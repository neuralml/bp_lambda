#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parent class for RNN's using BP(lambda)
RNNs also capable of using eligibility traces for learning: see Bellec et al. 2019
"""

import math
import torch
from torch.nn.parameter import Parameter
import sys
from functools import wraps
import os
import inspect

try:
    from .synthesiser import synthesiser
except:
    from synthesiser import synthesiser
import numpy as np

try:
    import dni
except:
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    if parentdir not in sys.path:
        sys.path.insert(0, parentdir) 
    import dni

class eprop_synthgrad(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, predict_last=False, 
                 apply_sg=False, synth_params=None, backprop=False, bptt=None, 
                 classification=False, batch_size=-1, evec_scale=None, 
                 forget_gate=None, seqlen=-1, use_etraces=True, alpha=None,
                 ):                
        """
        Parameters
        ----------
        input_size : external input dimension
        hidden_size : number of hidden RNN units
        output_size : size of targets/readout output layer
        predict_last : train model just to predict last timestep
        apply_sg : use synthetic gradients
        synth_params : parameters for the synthesiser network
        backprop : using backpropagation through time (BPTT)
        bptt : truncation size used in BPTT
        classification : regression or classification task
        batch_size : batch size
        evec_scale : scale eligibility traces (control extent of their propagation)
        forget_gate : used fixed forget gate for LSTM (improves bio-plausibility)
        seqlen : length of task
        use_etraces : use eligibility traces for learning (eprop; Bellec et al. 2019)
        alpha : membrane time constant for leaky RNN
        """
        
        super(eprop_synthgrad, self).__init__()
                
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.output_size = output_size
        self.predict_last = predict_last
        self.seqlen_whole = seqlen
        self.seqlen = seqlen if bptt is None else bptt
        self.use_etraces = use_etraces
        
        self.niterations = int(1000 / batch_size) #assume training set size is 1000
        
        self.optimizer = None
        
        if self.predict_last:    
            if bptt is not None:
                final_trunc_size = min(bptt, self.seqlen_whole)
                self.teacher_inds = [final_trunc_size - 1]  
            else:
                self.teacher_inds = [self.seqlen - 1]
        else:
            self.teacher_inds = np.arange(0, self.seqlen)
        
        if batch_size == -1:
            raise ValueError("rnn_bp_lambda model initialisation requires positive batch size")
        else:
            self.batch_size = batch_size
        self.timestep = -1
        
        self.backprop = backprop
        self.bptt = bptt
        self.evec_scale = evec_scale
                        
        if not self.backprop:
            assert self.bptt is None, "Backprop is false but bptt is {}.".format(self.bptt) \
                                    + "Check you know what you're doing."
        
        self.classification = classification
        
        if classification:
            self.get_output_derivative = self.get_output_derivative_nll
        else:
            self.get_output_derivative = self.get_output_derivative_mse
        
        self.apply_sg = apply_sg
        self.correct_when_learnt = False
        self.backprop_only_synth = False 
        self.trails_required = False
        self.record_etrace = False
        self.forget_gate_fixed = forget_gate
        self.record_grads = False
        self.task_start = None
        self.task_length = None
        
        self.weight_types = ['i', 'r', 'b']
         
        self.init_weights()
         
        if apply_sg:
            self.init_synth_stuff(synth_params)      
                        
        if self.backprop:
            self.hook_counter = -1
            self.trails_required = True
            self.accumalate_gradients = self.accumalate_gradients_withbackprop
            
            if self.apply_sg:
                self.backward_interface.requires_grad_()
        else:
            self.accumalate_gradients = self.accumalate_gradients_nobackprop
        
        self.init_buffers()
        
        self.target = None
        
        self.online_correction = False
        
        if not self.use_etraces:
            self.evec_scale = 0
            
        self.update_grads = self.update_grads_etraces if use_etraces else self.update_grads_cell_out

    
    def init_synth_stuff(self, synth_params):
        assert synth_params is not None, "Applying synthetic gradients, set synthesiser parameters!"   
        self.synth_cell_grad = synth_params['synth_cell_grad']
        print("initialising synthesiser")       
 
        if self.backprop:         
            if self.gate_size > self.hidden_size: #lstm case
                synth_inp_size = 2 * self.hidden_size if self.synth_cell_grad else self.hidden_size
            else:
                synth_inp_size = self.hidden_size
                        
            self.backward_interface = dni.BackwardInterface(
                dni.BasicSynthesizer(synth_inp_size, 
                                 n_hidden=synth_params['synth_nlayers'], 
                                 hidden_dim=synth_params['synth_nhid'])
                )
            self.control_synth_fibres(synth_params['nfibres'])
        else:
            if self.gate_size > self.hidden_size: #lstm case
                synth_inp_size = 2 * self.hidden_size
                synth_out_size = int(synth_inp_size/2) if self.use_etraces else synth_inp_size
            else:
                synth_inp_size = self.hidden_size
                synth_out_size = synth_inp_size
                
            self.synthesiser = synthesiser(synth_inp_size, 
                                       num_hidden_layers=synth_params['synth_nlayers'],
                                       hidden_size=synth_params['synth_nhid'], 
                                       alpha=synth_params['synth_alpha'], 
                                       gamma=synth_params['synth_gamma'], 
                                       lmbda=synth_params['synth_lambda'],
                                       opt_out_weight_only=synth_params['synth_opt_outputw_only'],
                                       batch_size=self.batch_size,
                                       bias=synth_params['bias'],
                                       alpha_learning=synth_params['alpha_learning'],
                                       nfibres=synth_params['nfibres'],
                                       cer_rate_reg=synth_params['cer_rate_reg'],
                                       output_size=synth_out_size) 
        if 'sg_weight' in synth_params.keys():
            self.sg_factor = synth_params['sg_weight']
        else:
            self.sg_factor = 0.1 #default choice in Jaderberg et al. 2017
      
        
        if 'backprop_only_synth' in synth_params.keys():
            self.backprop_only_synth = synth_params['backprop_only_synth']
            if self.backprop_only_synth and not self.backprop:
                sys.exit("backprop is false but backprop_only_synth is true. Doesn't make sense")
            if self.backprop_only_synth and not self.apply_sg:
                sys.exit("apply_sg if false but backprop_only_synth is true. Doesn't make sense")
            
    
    def control_synth_fibres(self, nfibres):
        if nfibres is None:
            return
        else:           
            def generate_synth_mask(input_shape, nfibres):
                mask = torch.ones(input_shape)
                for i in range(input_shape[0]):
                    connections = np.random.choice(input_shape[1], nfibres, replace=False)
                    mask[i, connections] = 0
                mask = mask.type(torch.bool)        
                return mask          
            
            def enforce_sparse(mask):
                def makezero(grad):
                    grad[mask] = 0          
                return makezero
            
            input_shape = self.backward_interface.synthesizer.input_trigger.weight.shape
            mask = generate_synth_mask(input_shape, nfibres)
            with torch.no_grad():
                self.backward_interface.synthesizer.input_trigger.weight[mask] = 0
                print("Initialised sparse synthesiser weights")
            
            self.backward_interface.synthesizer.input_trigger.weight.register_hook(enforce_sparse(mask))

    def join_hidden(self, hidden):
        hidden = torch.cat(hidden, dim=1)
        return hidden

    def split_hidden(self, hidden):
        (h, c) = hidden.chunk(2, dim=1)
        hidden = (h.contiguous(), c.contiguous())
        return hidden               

    def init_trails(self):
        raise NotImplementedError  

    @property
    def synth_inp_size(self):
        raise NotImplementedError  

    @property
    def gate_size(self):
        raise NotImplementedError  
        
    def init_weights(self):
        gate_size = self.gate_size  
        
        w_ih = Parameter(torch.Tensor(gate_size, self.input_size))
        w_hh = Parameter(torch.Tensor(gate_size, self.hidden_size))
        b = Parameter(torch.Tensor(gate_size))
        
        w_out = Parameter(torch.Tensor(self.output_size, self.hidden_size))
        b_out = Parameter(torch.Tensor(self.output_size))
        
        params = [w_ih, w_hh, b, w_out, b_out]
        param_names = ['w_ih', 'w_hh', 'b', 'w_out', 'b_out']

        for name, param in zip(param_names, params):
            setattr(self, name, param)
        
        self.requires_grad_(requires_grad=False)
        self.reset_parameters()    
    
    def init_buffers(self):
        shape = (self.batch_size, self.hidden_size)
                
        self.register_buffer('cell_state', torch.zeros(shape), persistent=False)
        self.register_buffer('out_state', torch.zeros(shape), persistent=False)
        
        if self.backprop:
            self.reset_trails(newseqlen=self.seqlen)
        
    def reset_trails(self, newseqlen=None):
        if newseqlen is not None:
            self.init_trails()
        else:
            for name, buffer in self.named_buffers():
                if 'trail' in name:
                    setattr(self, name, torch.zeros_like(buffer))
        
    def store_evecs(self):
        r"""If waiting for backpropagated learning signals then trail of eligibility traces 
            will be stored and correctly matched to them
        """
        raise NotImplementedError  
    
    def reset_buffers(self, new_batch_size=True):
        if new_batch_size:
            new_shape = (self.batch_size,) + self.cell_state.shape[1:]
            self.cell_state = self.cell_state.new_zeros(size=new_shape)   
            self.out_state = self.out_state.new_zeros(size=new_shape)           
            new_batch_size = False
            
        if self.training:
            for name, buffer in self.named_buffers():
                setattr(self, name, torch.zeros_like(buffer))

        elif not new_batch_size:
            self.cell_state = torch.zeros_like(self.cell_state)
            self.out_state = torch.zeros_like(self.out_state)

    # a small thing, but for a pytorch LSTM there's two bias variables for the hidden layer
    # in addition, they have a dependency so that the gradient get's multiplied by
    #two during optimisation. This means that to mimic the pytorch LSTM, we 
    #need to multiply the bias gradient by 2*2 = 4
    def correct_bias_grad(self):
        self.b.grad *= 2

    def reset_grads(self):   
        for weight in self.parameters():
            torch.nn.init.zeros_(weight.grad)
    
    def init_batch(self, input, target, latter_trunc=False, record_etrace=False, dataset_continue=False):
        if latter_trunc:
            prev_trunc_size = self.input.shape[1]
   
        self.input = input
        self.target = target
        self.latter_trunc = latter_trunc
        
        batch_size, seqlen = self.input.shape[:2]
        if batch_size == self.batch_size:
            first_init = False
            new_batch_size = False
        else:
            first_init = True
            new_batch_size = True
            self.batch_size = batch_size
            
        newseqlen = None
        if seqlen != self.seqlen:
           self.seqlen = seqlen
           newseqlen = seqlen
       
        if not latter_trunc and not dataset_continue:
            self.reset_buffers(new_batch_size=new_batch_size)
            if self.training and self.apply_sg and not self.backprop:
                self.synthesiser.reset_etraces()
                
            self.timestep_real = 0
            self.timestep = 0
        else:
            if self.backprop:
                self.timestep_real += prev_trunc_size
            else:
                self.timestep_real +=1 
            if dataset_continue:
                self.timestep = 0
            
        if self.trails_required and self.ls_available():
            self.hook_counter = -1  
            self.reset_trails(newseqlen)
                    
    def forward(self, input, hidden=None, target=None, latter_trunc=False, 
                noise_thrown=False, record_etrace=False):
        if hidden is not None:
            self.set_hidden(hidden, noise_thrown=noise_thrown, latter_trunc=latter_trunc)

        dataset_continue = hidden is not None

        self.init_batch(input, target, latter_trunc=latter_trunc, 
                        record_etrace=record_etrace, dataset_continue=dataset_continue)
        
        output = self.run_over_seq()
        if self.predict_last:
            output = output[:, -1, :]
                            
        return output, (self.out_state, self.cell_state)
    
    def forward_step(self):
        self.compute_states()                 
        self.compute_output()
        return self.readout
    
    def set_hidden(self, hidden, noise_thrown=False, latter_trunc=False):
        raise NotImplementedError  
             
    def run_over_seq(self):  
        output = self.input.new_zeros(size=(self.batch_size, self.seqlen, self.output_size))
        for t in range(self.seqlen):
            self.timestep = t
            self.input_t = self.input[:, t, :]
           
            output[:, t, :] = self.forward_step()
            if self.training:               
                self.accumalate_gradients()   
                
        if self.training and hasattr(self, 'synthesiser') and self.synthesiser.pred_cell_future:
            self.final_update_etraces()
        return output        
            
    def compute_states(self):        
        raise NotImplementedError  
    
    def backprop_hook(self, outstate_grad):
        raise NotImplementedError         

    def backprop_hook_outonly(self, outstate_grad):
        raise NotImplementedError

    def backprop_hook_cellonly(self, outstate_grad):
        raise NotImplementedError          
    
    def compute_output(self):
        self.readout = torch.mm(self.out_state, self.w_out.t()) + self.b_out              
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if 'synth' not in name:
                if name == 'w_out':
                    torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
                elif name == 'b_out':
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.w_out)
                    bound = 1 / math.sqrt(fan_in)
                    torch.nn.init.uniform_(weight, -bound, bound)   
                elif name == 'b':
                    with torch.no_grad():
                        weight.data = torch.zeros_like(weight).uniform_(-stdv, stdv) + torch.zeros_like(weight).uniform_(-stdv, stdv)
                else:                
                    torch.nn.init.uniform_(weight, -stdv, stdv) 
                weight.grad = torch.zeros_like(weight)
                        
    def accumalate_gradients_nobackprop(self):
        self.update_traces()
        if self.ls_available():
            self.update_grads()
            if self.opt_each_trunc:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
    def accumalate_gradients_withbackprop(self):
        raise NotImplementedError
                   
    def ls_available(self):
        ls_available = False
        if not self.training:
            ls_available = False
        elif self.apply_sg:
            ls_available = True
        elif self.target is None:
            ls_available = False
        elif self.backprop and self.target is not None:
            if not self.backprop_only_synth:
                ls_available = True         
        elif self.timestep in self.teacher_inds:
            ls_available = True 
            
        return ls_available
            
    def update_readout_grads(self):
        if self.predict_last:
            if self.target is not None and self.timestep in self.teacher_inds:
                if self.classification and self.target.ndim > 1:
                    target = self.target[:, -1]
                else:
                    target = self.target
            else: 
                return               
        else:
            if self.target is not None and self.timestep in self.teacher_inds:
                if self.target.shape[1] < self.input.shape[1]: #early target shape...
                    target_buffer = self.input.shape[1] - self.target.shape[1]
                    if self.timestep >= target_buffer:
                        target = self.target[:, self.timestep-target_buffer]
                    else:
                        return
                else:
                    target = self.target[:, self.timestep]
            else:
                return
        
        dL_do = self.get_output_derivative(target)
        
        w_out_grad = torch.bmm(dL_do[:,:, None], self.out_state[:, None, :])
        b_out_grad = dL_do
        
        if self.target is not None and self.target.ndim > 1:
            length = self.target.shape[1]
            if self.task_start is not None:
                if self.bptt is None:
                    length -= self.task_start
                elif self.task_length is not None: #for copy repeat task
                    length = self.task_length - self.task_start
                    
            scale = 1/length
        else:
            scale = 1

        self.w_out.grad += torch.sum(w_out_grad, dim=0) * scale
        self.b_out.grad += torch.sum(b_out_grad, dim=0) * scale

        return dL_do 
    
    def get_output_derivative_mse(self, target):
        return 1/self.batch_size * (self.readout - target)

    def get_output_derivative_nll(self, target):
        target_one_hot = self.one_hot(target)
        return 1/self.batch_size * (torch.softmax(self.readout, dim=1) - target_one_hot)
   
    def get_output_derivative_nll_raw(self, target):
        return 1/(self.batch_size*self.output_size) * (torch.sigmoid(self.readout) - target)
 
    def one_hot(self, batch):
        one_hot = batch.new_zeros(size=(batch.shape[0], self.output_size))
        one_hot[torch.arange(0, batch.shape[0]), batch] = 1
        return one_hot

    def get_learning_signal(self, imm_grad):  
        grad = imm_grad if imm_grad is not None else 0
        if self.apply_sg:
            hiddtohidd_grad = self.get_hidtohid_grad()
            first_forward = False
            final_step = False
            if self.timestep == 0:
                first_forward = True
            elif self.timestep == self.seqlen -1 and self.target is not None:
                final_step = True

            synth_grad = self.synthesiser(self.out_state, imm_grad, hiddtohidd_grad, 
                             first_forward=first_forward, 
                             final_step=final_step, verbose=False)
                       
            grad += self.sg_factor * synth_grad
                
        if self.record_grads:
            self.recorded_grads[self.nepoch, self.iteration_number, :, self.timestep] = grad

        return grad
          
                            
    def update_grads_etraces(self):
        raise NotImplementedError  
    
    def update_grads_cell_out(self):
        raise NotImplementedError
    
    def update_grads_evecs(self):
        raise NotImplementedError  
                
    def get_hidtohid_grad(self):       
        raise NotImplementedError  
        
    def get_hidtoweight_grad_direct(self, param_type):
        raise NotImplementedError  
    
    def get_hidtoweight_grad_evec(self, param_type):
        raise NotImplementedError  
                    
    def record_grads_(self, record=True, first_inst=True, batch_size=None, 
                      seqlen=None, epochs=None, nepoch=None,):
        self.record_grads = record
        self.nepoch = nepoch
        
        if not self.use_etraces and self.gate_size > self.hidden_size:
            grad_size = self.hidden_size * 2
        else:
            grad_size = self.hidden_size
                
        if first_inst:
            self.iteration_number = -1
            self.recorded_grads = torch.zeros(epochs, self.niterations, batch_size, 
                                              seqlen, grad_size)   
               
    def update_traces(self):
        raise NotImplementedError  
        
    def wrap_predlast_train(self, func):
        @wraps(func)
        def wrapper(training=True):
            self.predict_last = training
            return func(training)
        return wrapper
        
    def wrap_predlast_val(self, func):
        @wraps(func)
        def wrapper(training=True):
            self.predict_last = not training
            return func(training)
        return wrapper
