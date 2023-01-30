#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
standard RNN with bp_lambda::
"""

import torch

try:
    from eprop_synthgrad import eprop_synthgrad
except:
    from .eprop_synthgrad import eprop_synthgrad

class rnn_bp_lambda(eprop_synthgrad):
    def __init__(self, non_lin='identity', *args, **kwargs):
        super(rnn_bp_lambda, self).__init__(*args, **kwargs)
                   
        self.alpha = kwargs['alpha']
        if non_lin == 'tanh':
            self.f = torch.tanh
            self.f_prime = lambda x: 1 - torch.tanh(x)**2
        elif non_lin == 'sigmoid':
            self.f = torch.sigmoid
            self.f_prime = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
        elif non_lin == 'relu':
            self.f = torch.relu
            self.f_prime = lambda x: torch.sign(x).clamp(0)
        elif non_lin == 'identity':
            self.f = lambda x: x * 1
            self.f_prime = lambda x: torch.ones_like(x)

        if self.alpha > 0 and not self.backprop:
            self.init_evecs()
            self.get_hidtoweight_grad = self.get_hidtoweight_grad_evec
        else:
            self.get_hidtoweight_grad = self.get_hidtoweight_grad_direct   
    
        assert self.use_etraces, 'RNN module currently requires use of etraces'

    @property
    def synth_inp_size(self):
        return self.hidden_size 

    @property
    def gate_size(self):
        return self.hidden_size       
        
    def init_evecs(self):
        shape = (self.batch_size, self.hidden_size)
        for i, weight in enumerate(self.parameters()):
            if i > 2:
                continue
            s = [(self.input_size,), (self.hidden_size, ), ()][i]
            
            name = self.weight_types[i]          
            name += '_evec'
            self.register_buffer(name, torch.zeros(shape + s), persistent=False)
    
    def init_trails(self,):
        shape = (self.batch_size, self.hidden_size)
        trail_shape = (self.seqlen,) + shape
        self.register_buffer('cell_state_trail', torch.zeros(trail_shape), persistent=False)
        
        for i, weight in enumerate(self.parameters()):
            if i > 2:
                continue
            s = [(self.input_size,), (self.hidden_size, ), ()][i]
            
            name = self.weight_types[i]          
            name += '_evec'
            
            self.register_buffer(name, torch.zeros(shape + s), persistent=False)
            
            trail_length = self.seqlen if self.bptt is None else self.bptt
            trail_shape = (trail_length,) + shape + s
            name_trail = name + '_trail'
            self.register_buffer(name_trail, torch.zeros(trail_shape),
                                 persistent=False)
            
        if self.use_etraces:
            trail_shape = (trail_length,) + shape
            self.register_buffer('out_cell_grad_trail', torch.zeros(trail_shape),
                             persistent=False)        
            
        
    def store_evecs(self):
        r"""If waiting for backpropagated learning signals then trail of eligibility traces 
            will be stored and correctly matched to them
        """
        hook_counter = self.seqlen - self.timestep - 1

        for weight_type in self.weight_types:
            name = weight_type
            name += '_evec'
            evec_now = getattr(self, name)
            

            evec_trail = getattr(self, name + '_trail')           
            evec_trail[hook_counter] = evec_now
            
        out_cell_grad = self.f_prime(self.cell_state)  
        self.out_cell_grad_trail[hook_counter] = out_cell_grad
    
    def set_hidden(self, hidden, noise_thrown=False, latter_trunc=False):
        if self.train and self.backprop and latter_trunc:
            if self.apply_sg:
                self.out_state = self.backward_interface.make_trigger(hidden[0])  
                self.cell_state = hidden[1]
            else:
                self.out_state, self.cell_state = hidden         
                   
    def compute_states(self):        
        self.cell_state_old = self.cell_state
        self.out_state_old = self.out_state
        
        self.cell_state = torch.mm(self.input_t, self.w_ih.t()) \
                    + torch.mm(self.out_state, self.w_hh.t()) + self.b  
        
       
        self.out_state = self.f(self.cell_state)   
        
        if self.backprop and self.ls_available():               
            self.out_state.requires_grad_()
            self.out_state.register_hook(self.backprop_hook)
            
    def backprop_hook(self, outstate_grad):
        self.hook_counter += 1
        
        if self.record_grads:
            if self.bptt is not None:
                timestep = self.seqlen - self.hook_counter - 1 + self.timestep_real
            elif self.seqlen == 1:
                timestep = self.timestep_real
            else:
                timestep = self.seqlen - self.hook_counter - 1
                
            self.recorded_grads[self.nepoch, self.iteration_number, :, timestep] = outstate_grad

        out_cell_grad_gen = self.out_cell_grad_trail[self.hook_counter]

        for i, weight in enumerate([self.w_ih, self.w_hh, self.b]):
            if i < 2:
                out_cell_grad = out_cell_grad_gen[:, :, None]
                ls = outstate_grad[:, :, None]               
            else:
                out_cell_grad = out_cell_grad_gen
                ls = outstate_grad
            
            evec_trail = getattr(self, self.weight_types[i] + '_evec_trail')            
            etrace = evec_trail[self.hook_counter] * out_cell_grad
            
            grad = torch.sum(etrace * ls, dim=0)
                        
            if i == 2:
                grad *= 2
            weight.grad += grad   
                
            
    def accumalate_gradients_withbackprop(self):
        self.update_traces()
        if self.ls_available():
            pred = self.update_readout_grads()
    
            self.store_evecs()
            if self.apply_sg and self.timestep == self.seqlen - 1:
                if self.synth_cell_grad:
                    h = self.join_hidden((self.out_state, self.cell_state))
                    self.backward_interface.backward(h, factor=self.sg_factor)     
                else:
                    self.backward_interface.backward(self.out_state, factor=self.sg_factor)   
                   
         

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
        diff = self.update_readout_grads()     
        imm_grad = torch.mm(diff, self.w_out) if diff is not None else None     
        learning_signal = self.get_learning_signal(imm_grad)   

        params = ['i', 'r', 'b']
        for i, weight in enumerate([self.w_ih, self.w_hh, self.b]):
            ls = learning_signal[:, :, None] if i < 2 else learning_signal
            dhid_dweight = self.get_hidtoweight_grad(params[i])
            grad = torch.sum(dhid_dweight * ls, dim=0)      
            weight.grad += grad    
                
    def get_hidtohid_grad(self):       
        grad = self.f_prime(self.cell_state[:, :, None]) * self.w_hh[None, :] 
        return grad
    
    def get_hidtoweight_grad_direct(self, param_type):
        dout_dcell = self.f_prime(self.cell_state)
        
        if param_type == 'i':
            dout_dweight = torch.bmm(dout_dcell[:, :, None], 
                                      self.input_t[:, None, :])
        elif param_type == 'r':
            dout_dweight = torch.bmm(dout_dcell[:, :, None], 
                                      self.out_state_old[:, None, :])
        elif param_type == 'b':
            dout_dweight = dout_dcell
            
        return dout_dweight
    
    def get_hidtoweight_grad_evec(self, param_type):
        dout_dcell = self.f_prime(self.cell_state)
        
        if param_type in ['i', 'r']:
            dout_dcell = dout_dcell[:, :, None]
                    
        evec = getattr(self, param_type + '_evec')            
        etrace = evec * dout_dcell
            
        return etrace
            
    def update_traces(self):
        r"""update the eligibility traces
        """    
        if self.alpha == 0:
            return 
        
        post_term = self.input_t[:, None, :].repeat((1, self.hidden_size, 1))
        self.i_evec = self.i_evec * self.alpha + post_term
                              
        post_term = self.out_state_old[:, None, :].repeat((1, self.hidden_size, 1))                     
        self.r_evec = self.r_evec * self.alpha + post_term
            
        if self.bias:
            self.b_evec = self.b_evec * self.alpha + torch.ones_like(self.b_evec)    
            
