#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM with bp_lambda:
"""

import torch

try:
    from eprop_synthgrad import eprop_synthgrad
except:
    from .eprop_synthgrad import eprop_synthgrad

class lstm_bp_lambda(eprop_synthgrad):
    def __init__(self, forget_gate=None, *args, **kwargs):
        super(lstm_bp_lambda, self).__init__(*args, **kwargs)              
 
        self.forget_gate_fixed = forget_gate
        if not self.backprop:
            self.init_evecs()
            
        self.register_buffer('out_cell_grad', torch.zeros_like(self.cell_state), persistent=False)

        if self.apply_sg and not self.backprop:
            if self.use_etraces: #predict dL_dc
                self.synthesiser.pred_cell_future = True 
                self.register_buffer('imm_grad', torch.zeros_like(self.cell_state), persistent=False)
                self.register_buffer('out_gate', torch.zeros_like(self.cell_state), persistent=False)
                
            self.synth_cell_grad = not self.use_etraces

    @property
    def gate_size(self):
        return self.hidden_size * 4     
        
    def init_evecs(self):
        shape = (self.batch_size, self.hidden_size)
        for gate in ['inp', 'forget', 'out', 'cand']:
            for weight_type, s in zip(['r', 'i', 'b'], [(self.hidden_size,), 
                                                     (self.input_size,), ()]):
                
                name = gate + weight_type
                if gate == 'out':                       
                    name += '_etrace'
                else:
                    name += '_evec'
                    
                self.register_buffer(name, torch.zeros(shape + s), persistent=False)
    
    
    def init_trails(self,):
        shape = (self.batch_size, self.hidden_size)
        trail_shape = (self.seqlen,) + shape
        device = self.w_hh.device
        for gate in ['inp', 'forget', 'out', 'cand']:
            for weight_type, s in zip(['r', 'i', 'b'], [(self.hidden_size,), 
                                                     (self.input_size,), ()]):
                
                name = gate + weight_type
                if gate == 'out':                       
                    name += '_etrace'
                else:
                    name += '_evec'
                
                name_trail = name + '_trail'
                
                self.register_buffer(name, torch.zeros(shape + s, device=device), persistent=False)
                if self.trails_required:
                    trail_length = self.seqlen if self.bptt is None else self.bptt
                    trail_shape = (trail_length,) + shape + s
                    self.register_buffer(name_trail, torch.zeros(trail_shape, device=device),
                                         persistent=False)
                
        if self.use_etraces:
            trail_shape = (trail_length,) + shape 
            self.register_buffer('out_cell_grad_trail', torch.zeros(trail_shape, device=device),
                                 persistent=False)   
            
        
    def store_evecs(self):
        hook_counter = self.seqlen - self.timestep - 1
        
        for gate in ['inp', 'forget', 'out', 'cand']:
            for weight_type in ['r', 'i', 'b']:
                name = gate + weight_type
                if gate == 'out':                       
                    name += '_etrace'
                else:
                    name += '_evec'
                evec_now = getattr(self, name)
                evec_trail = getattr(self, name + '_trail')                
                evec_trail[hook_counter] = evec_now
        
        if self.use_etraces:
            self.out_cell_grad_trail[hook_counter] = self.out_cell_grad
                                    
    def set_hidden(self, hidden, noise_thrown=False, latter_trunc=False):
        if self.train and self.backprop and self.apply_sg:
            if self.synth_cell_grad:
                hidden = self.join_hidden(hidden)
                hidden = self.backward_interface.make_trigger(hidden)
                hidden = self.split_hidden(hidden)       
                self.out_state, self.cell_state = hidden
            else:               
                self.out_state = self.backward_interface.make_trigger(hidden[0])  
                self.cell_state = hidden[1]
        else:
            self.out_state, self.cell_state = hidden
             
    
    def compute_gates(self):        
        gates = torch.mm(self.input_t, self.w_ih.t()) \
                    + torch.mm(self.out_state, self.w_hh.t()) + self.b   

        ingate, forgetgate, candgate, outgate = gates.chunk(4, 1)
        
        if self.forget_gate_fixed is not None:
            forgetgate = self.forget_gate_fixed * torch.ones_like(forgetgate)
            
        if self.apply_sg and self.use_etraces:
            self.out_gate_old = self.out_gate.clone()
        
        self.inp_gate = torch.sigmoid(ingate)
        self.forget_gate = torch.sigmoid(forgetgate)
        self.out_gate = torch.sigmoid(outgate)
        self.cand_gate = torch.tanh(candgate)             
    
    def compute_states(self):
        self.compute_gates()
        self.cell_state_old = self.cell_state.clone()
        self.out_state_old = self.out_state.clone()
                
                
        self.cell_state = (self.forget_gate * self.cell_state) + (self.inp_gate * self.cand_gate)  
        
        if self.backprop and self.ls_available() and not self.use_etraces:
            self.cell_state.requires_grad_()
            self.cell_state.register_hook(self.backprop_hook_cellonly)             
        
        self.out_state = self.out_gate * torch.tanh(self.cell_state)     
        
        if self.backprop and self.ls_available():
            self.out_state.requires_grad_()
            if self.use_etraces:                
                self.out_state.register_hook(self.backprop_hook)    
            else:
                self.out_state.register_hook(self.backprop_hook_outonly) 
        
    def backprop_hook(self, outstate_grad):        
                      
        self.hook_counter += 1
        assert self.hook_counter <= self.seqlen
            
        if self.record_grads:
            if self.seqlen == 1:
                timestep = self.timestep_real
            else:
                timestep = self.seqlen - self.hook_counter - 1
                
            self.recorded_grads[self.nepoch, self.iteration_number, :, timestep] = outstate_grad
        
        params = ['i', 'r', 'b']
        out_cell_grad_gen = self.out_cell_grad_trail[self.hook_counter]
        for i, weight in enumerate([self.w_ih, self.w_hh, self.b]):            
            if i < 2:
                out_cell_grad = out_cell_grad_gen[:, :, None]
                ls = outstate_grad[:, :, None]               
            else:
                out_cell_grad = out_cell_grad_gen
                ls = outstate_grad
            
            for j, gate in enumerate(['inp', 'forget', 'cand', 'out']):
                if gate == 'out':
                    etrace_trail = getattr(self, gate + params[i] + '_etrace_trail')  
                    etrace = etrace_trail[self.hook_counter] 
                else:
                    evec_trail = getattr(self, gate + params[i] + '_evec_trail')
                    etrace = evec_trail[self.hook_counter] * out_cell_grad
                
                grad = torch.sum(etrace * ls, dim=0)

                if i == 2:
                    grad *= 2
                    
                weight.grad[j*self.hidden_size: (j+1)*self.hidden_size] += grad  
                
    
    def backprop_hook_outonly(self, outstate_grad):        
        self.hook_counter += 1
        
        assert self.hook_counter <= self.seqlen
        
        if self.record_grads:
            if self.seqlen == 1:
                timestep = self.timestep_real
            else:
                timestep = self.seqlen - self.hook_counter - 1
                
            self.recorded_grads[self.nepoch, self.iteration_number, :, timestep, :self.hidden_size] = outstate_grad
        
        params = ['i', 'r', 'b']
        for i, weight in enumerate([self.w_ih, self.w_hh, self.b]):            
            if i < 2:
                ls = outstate_grad[:, :, None]               
            else:
                ls = outstate_grad
            
            gate = 'out'
            etrace_trail = getattr(self, gate + params[i] + '_etrace_trail')  
            etrace = etrace_trail[self.hook_counter] 
                
            grad = torch.sum(etrace * ls, dim=0)
            
            if i == 2:
                grad *= 2
            weight.grad[3*self.hidden_size:] += grad 
                    
    def backprop_hook_cellonly(self, cellstate_grad):               
        assert self.hook_counter <= self.seqlen
        
        if self.record_grads:
            if self.seqlen == 1:
                timestep = self.timestep_real
            else:
                timestep = self.seqlen - self.hook_counter - 1
                
            self.recorded_grads[self.nepoch, self.iteration_number, :, timestep, self.hidden_size:] = cellstate_grad
        
        params = ['i', 'r', 'b']
        for i, weight in enumerate([self.w_ih, self.w_hh, self.b]):            
            if i < 2:
                ls = cellstate_grad[:, :, None]               
            else:

                ls = cellstate_grad
            
            for j, gate in enumerate(['inp', 'forget', 'cand']):
                evec_trail = getattr(self, gate + params[i] + '_evec_trail')
                evec = evec_trail[self.hook_counter]
                
                grad = torch.sum(evec * ls, dim=0)
                
                if i == 2:
                    grad *= 2
                weight.grad[j*self.hidden_size: (j+1)*self.hidden_size] += grad  
        
    def accumalate_gradients_withbackprop(self):
        with torch.no_grad():
            self.update_traces()
        if self.ls_available():
            pred = self.update_readout_grads()
            self.store_evecs()   

            if self.apply_sg:
                last_trunc = self.timestep_real + self.input.shape[1] == self.seqlen_whole
                if last_trunc: #define sg as 0
                    return
                elif self.timestep == self.seqlen - 1:
                    synth_inp = self.join_hidden((self.out_state, self.cell_state)) if self.synth_cell_grad else self.out_state
                    self.backward_interface.backward(synth_inp, factor=self.sg_factor)          
         

    def get_learning_signal(self, imm_grad,):  
        grad = imm_grad if imm_grad is not None else 0
        if self.apply_sg:
            if self.use_etraces:
                hiddtohidd_grad, dc_dh = self.get_celltocell_grad_solo()
                self.imm_grad_old = self.imm_grad
                self.imm_grad = imm_grad #remember, estimating dE/dc
                if self.imm_grad_old is not None:
                    imm_grad = self.imm_grad_old * self.out_cell_grad_old
                    grad = self.imm_grad_old
                else:
                    imm_grad = None
                    grad = 0
               
                synth_inp = torch.cat((self.out_state_old, self.cell_state_old), dim=1)                 
            else:
                hiddtohidd_grad = self.get_hidtohid_grad_all()
                synth_inp = torch.cat((self.out_state, self.cell_state), dim=1)

            first_forward = False
            final_step = False
            if self.timestep == 0:
                first_forward = True
            elif self.timestep == self.seqlen -1 and self.target is not None and not self.use_etraces:
                final_step = True

            synth_grad = self.synthesiser(synth_inp, imm_grad, hiddtohidd_grad, 
                             first_forward=first_forward, 
                             final_step=final_step, verbose=False)       

            if self.use_etraces:
                synth_grad = torch.bmm(synth_grad[:, None,], dc_dh).reshape(synth_grad.shape)

            grad += self.sg_factor * synth_grad
                
        if self.record_grads:
            trec = self.timestep
            if self.apply_sg and self.use_etraces:
                trec -= 1
            if trec >= 0:
                self.recorded_grads[self.nepoch, self.iteration_number, :, trec] = grad

        return grad
          
    def update_grads_etraces(self):
        diff = self.update_readout_grads()     
        lstm_imm_grad = torch.mm(diff, self.w_out) if diff is not None else None
        out_cell_grad_gen = self.out_cell_grad
            
        lstm_learning_signal = self.get_learning_signal(lstm_imm_grad,)    
        
        if self.apply_sg:
            out_cell_grad_gen = self.out_cell_grad_old
            if self.timestep == self.seqlen - 1:
                self.imm_grad_final = lstm_imm_grad
            if self.timestep == 0:
                return

        params = ['i', 'r', 'b']
        for i, weight in enumerate([self.w_ih, self.w_hh, self.b]):
            if i < 2:
                out_cell_grad = out_cell_grad_gen[:, :, None]
                ls = lstm_learning_signal[:, :, None]               
            else:
                out_cell_grad = out_cell_grad_gen
                ls = lstm_learning_signal
            for j, gate in enumerate(['inp', 'forget', 'cand', 'out']):
                suff = '_etrace' if gate == 'out' else '_evec'
                name = gate + params[i] + suff
                if self.apply_sg:
                    name += '_old'
                etrace = getattr(self, name) 
                if gate != 'out':
                    etrace = etrace * out_cell_grad
                    
                grad = torch.sum(etrace * ls, dim=0)    
                if i == 2:
                    grad *= 2
                
                weight.grad[j*self.hidden_size: (j+1)*self.hidden_size] += grad    
                
    def final_update_etraces(self):
        #update the synthesiser..
        lstm_learning_signal = self.imm_grad_final
        if lstm_learning_signal is None:
            return
        imm_grad = lstm_learning_signal * self.out_cell_grad if lstm_learning_signal is not None else None
        if self.record_grads:
            self.recorded_grads[self.nepoch, self.iteration_number, :, -1] = lstm_learning_signal
        
        synth_inp = self.out_state
        hiddtohidd_grad = None
        _ = self.synthesiser(synth_inp, imm_grad, hiddtohidd_grad, 
                     first_forward=False, 
                     final_step=True, verbose=False)
        
        out_cell_grad_gen = self.out_cell_grad
        
        params = ['i', 'r', 'b']
        for i, weight in enumerate([self.w_ih, self.w_hh, self.b]):
            if i < 2:
                out_cell_grad = out_cell_grad_gen[:, :, None]
                ls = lstm_learning_signal[:, :, None]               
            else:
                out_cell_grad = out_cell_grad_gen
                ls = lstm_learning_signal
                
            for j, gate in enumerate(['inp', 'forget', 'cand', 'out']):
                suff = '_etrace' if gate == 'out' else '_evec'
                name = gate + params[i] + suff
                etrace = getattr(self, name) 
                if gate != 'out':
                    etrace = etrace * out_cell_grad
                                
                grad = torch.sum(etrace * ls, dim=0)          
                if i == 2:
                    grad *= 2
                
                weight.grad[j*self.hidden_size: (j+1)*self.hidden_size] += grad  
                
    def update_grads_cell_out(self,):
        diff = self.update_readout_grads()     
        out_cell_grad_gen = self.out_cell_grad
        
        if diff is not None:
            lstm_imm_grad_out = torch.mm(diff, self.w_out)   
            lstm_imm_grad_cell = lstm_imm_grad_out * out_cell_grad_gen 
            lstm_imm_grad = torch.cat((lstm_imm_grad_out, lstm_imm_grad_cell), dim=1)
        else:
            lstm_imm_grad = None
        
        lstm_learning_signal = self.get_learning_signal(lstm_imm_grad,) 
        
        if isinstance(lstm_learning_signal, int): #learning signal is 0..
            return

        outstate_grad = lstm_learning_signal[:, :self.hidden_size] 
        cellstate_grad = lstm_learning_signal[:, self.hidden_size:]   

        params = ['i', 'r', 'b']
        for i, weight in enumerate([self.w_ih, self.w_hh, self.b]):
            if i < 2:
                ls = lstm_learning_signal[:, :, None]               
            else:
                ls = lstm_learning_signal

            for j, gate in enumerate(['inp', 'forget', 'cand', 'out']):
                if gate == 'out':
                    etrace = getattr(self, gate + params[i] + '_etrace')                   
                else:
                    etrace = getattr(self, gate + params[i] + '_evec')
                                
                state_grad = outstate_grad if gate == 'out' else cellstate_grad
                ls = state_grad[:, :, None] if i < 2 else state_grad
                                
                grad = torch.sum(etrace * ls, dim=0)   
                if i == 2:
                    grad *= 2
                
                weight.grad[j*self.hidden_size: (j+1)*self.hidden_size] += grad    
                
    def get_lstm_grad(self, out_cell_grad_gen):
        w_hh_inp, w_hh_forget, w_hh_cand, w_hh_output = self.w_hh[None,:,:].chunk(4, dim=1)
        
        grad = self.inp_post_term[:,:,None] * w_hh_inp
        grad += self.forget_post_term[:,:,None] * w_hh_forget
        grad += self.cand_post_term[:,:,None] * w_hh_cand
        
        
        grad = out_cell_grad_gen[:, :, None] * grad #dh/dc * dc/dh
        
        grad += self.output_post_term[:,:,None] * w_hh_output
        
        return grad
                
    def get_celltocell_grad_solo(self,):       
        r = torch.arange(self.hidden_size)
        w_hh_inp, w_hh_forget, w_hh_cand, w_hh_output = self.w_hh[None,:,:].chunk(4, dim=1)
               
        dc_dh = self.inp_post_term[:,:,None] * w_hh_inp
        dc_dh += self.forget_post_term[:,:,None] * w_hh_forget
        dc_dh += self.cand_post_term[:,:,None] * w_hh_cand # now dc_dh
        
        
        dh_dc = torch.zeros_like(dc_dh)
        dh_dc[:, r, r] = self.out_cell_grad_old #dh_dc
        
        dc_dc = torch.bmm(dc_dh, dh_dc) #dh_dc x dc_dh
        dc_dc[:, r, r] += self.forget_gate #dc_dc
        
        return dc_dc, dc_dh
    
    def get_hidtohid_grad_all(self,):  
        r = torch.arange(self.hidden_size)
        w_hh_inp, w_hh_forget, w_hh_cand, w_hh_output = self.w_hh[None,:,:].chunk(4, dim=1)
             
        dh_dh = self.output_post_term[:,:,None] * w_hh_output #dh_dh
        dh_dc = torch.zeros_like(dh_dh)
        dh_dc[:, r, r] = self.out_cell_grad #dh_dc
        
        dc_dh = self.inp_post_term[:,:,None] * w_hh_inp
        dc_dh += self.forget_post_term[:,:,None] * w_hh_forget
        dc_dh += self.cand_post_term[:,:,None] * w_hh_cand # now dc_dh
        
        dc_dc = torch.zeros_like(dc_dh)
        dc_dc[:, r, r] = self.forget_gate #dc_dc
        
        second_row = torch.cat((dc_dh, dc_dc), dim=2) #dc/dx
        
        first_row = torch.bmm(dh_dc, second_row) #dh/dx
        
        grad = torch.cat((first_row, second_row), dim=1)
            
        return grad
            
            
    def update_traces(self):
        if self.apply_sg and not self.backprop and self.use_etraces:
            self.record_old_etraces()
        if self.evec_scale is not None:
            self.enfore_evec_decay()
        self.update_inp_evecs()
        self.update_forget_evecs()
        self.update_output_trace()
        self.update_cand_evecs()     
        self.out_cell_grad_old = self.out_cell_grad.clone()
        self.out_cell_grad = self.out_gate * (1 - torch.tanh(self.cell_state)**2)
                
    def update_inp_evecs(self):
        #batch multiply post by pre, where post = (batch_size, hidden_size, 1), pre = (batch_size, 1, hidden_size)
        post_synapse_term = (self.cand_gate * self.inp_gate * (1 - self.inp_gate))
        self.inp_post_term = post_synapse_term
        
        self.inpr_evec = self.inpr_evec * self.forget_gate[:, :, None] \
                            + torch.bmm(post_synapse_term[:, :, None], 
                                        self.out_state_old[:, None, :])
        self.inpi_evec = self.inpi_evec * self.forget_gate[:, :, None] \
                            + torch.bmm(post_synapse_term[:, :, None], 
                                        self.input_t[:, None, :])
        self.inpb_evec = self.inpb_evec * self.forget_gate \
                            + post_synapse_term
        
    def update_forget_evecs(self):
        #batch multiply post by pre, where post = (batch_size, hidden_size, 1), pre = (batch_size, 1, hidden_size)
        post_synapse_term = (self.cell_state_old * self.forget_gate * (1 - self.forget_gate))
        self.forget_post_term = post_synapse_term
        
        self.forgetr_evec = self.forgetr_evec * self.forget_gate[:, :, None] \
                            + torch.bmm(post_synapse_term[:, :, None], 
                                        self.out_state_old[:, None, :])
        
        self.forgeti_evec = self.forgeti_evec * self.forget_gate[:, :, None] \
                            + torch.bmm(post_synapse_term[:, :, None], 
                                        self.input_t[:, None, :])
        #print("input forget vec now:", self.forgeti_evec)
        self.forgetb_evec = self.forgetb_evec * self.forget_gate \
                            + post_synapse_term
 
    
    def update_output_trace(self):
        #batch multiply post by pre, where post = (batch_size, hidden_size, 1), pre = (batch_size, 1, hidden_size)
        #post_synapse_term = (self.cell_state * self.out_state * (1 - self.out_state))  
        post_synapse_term = (torch.tanh(self.cell_state) * self.out_gate  \
                             * (1 - self.out_gate))  
        self.output_post_term = post_synapse_term #save for learning signal later..
        
        self.outr_etrace = torch.bmm(post_synapse_term[:, :, None], 
                                      self.out_state_old[:, None, :])
        self.outi_etrace = torch.bmm(post_synapse_term[:, :, None], 
                                      self.input_t[:, None, :])
        self.outb_etrace = post_synapse_term
        
        #print("output b etrace:", self.outb_etrace.reshape(-1)[:4])

    def update_cand_evecs(self):
        #batch multiply post by pre, where post = (batch_size, hidden_size, 1), pre = (batch_size, 1, hidden_size)
        post_synapse_term = (self.inp_gate * (1 - self.cand_gate**2))
        self.cand_post_term = post_synapse_term
        
        self.candr_evec = self.candr_evec * self.forget_gate[:, :, None] \
                            + torch.bmm(post_synapse_term[:, :, None], 
                                        self.out_state_old[:, None, :])
        self.candi_evec = self.candi_evec * self.forget_gate[:, :, None] \
                            + torch.bmm(post_synapse_term[:, :, None], 
                                        self.input_t[:, None, :])
        self.candb_evec = self.candb_evec * self.forget_gate \
                            + post_synapse_term

    def record_old_etraces(self):
        params = ['i', 'r', 'b']
        for i in range(3):
            for j, gate in enumerate(['inp', 'forget', 'cand', 'out']):
                suff = '_etrace' if gate == 'out' else '_evec'
                name = gate + params[i] + suff
                etrace = getattr(self, name).clone()
                
                setattr(self, name + '_old', etrace)
                    
    def enfore_evec_decay(self):
        for gate in ['inp', 'forget', 'cand']:
            for weight_type in ['r', 'i', 'b']:
                evec = getattr(self, gate + weight_type + '_evec')
                evec *= self.evec_scale 
                
