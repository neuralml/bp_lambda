#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The synthesiser: given RNN state it predicts the future error gradient
synthesiser learns using BP(lambda)
"""

import sys, os
import torch
from torch.nn import init
import numpy as np

class synthesiser(torch.nn.Module):
    def __init__(self, input_size,  num_hidden_layers= 0, hidden_size=None, 
                 bias=True, alpha=0.01, gamma=1, lmbda=0.7, opt_out_weight_only=False,
                 batch_size=-1, cer_rate_reg=0, alpha_learning=False, nfibres=None,
                 output_size=None):
            """
            Parameters
            ----------
            input_size : input size
            num_hidden_layers : number of hidden layers
            hidden_size : size of hidden layers
            bias : use bias weights
            alpha : learning rate if 'alpha_learning' is true
            gamma : discount factor
            lmbda : lambda term which controls eligibility trace propagation
            opt_out_weight_only : only optimise synthesiser output weights (can be faster/computationally cheaper)
            batch_size : batch size
            cer_rate_reg : l1 regression on synthesiser activity
            alpha_learning : update synthesiser paramters at each timestep (without special optimiser/adam)
            nfibres : sparsify input weight connectivity (inspired by cerebellar mossy fibre architecture)
            output_size : output size
            """
            super(synthesiser, self).__init__()
                        
            if num_hidden_layers is None:
                num_hidden_layers = 0
            
            if num_hidden_layers > 0 and hidden_size is None:
                print("Setting hidden size to match input size ({})".format(input_size))
                hidden_size = input_size
                               
            self.input_size = input_size
            self.batch_size = None
            self.num_hidden_layers = num_hidden_layers
            self.hidden_size = hidden_size
            self.alpha = alpha
            self.gamma = gamma
            self.lmbda = lmbda
            self.batch_size = batch_size
            self.alpha_learning = alpha_learning
            self.nfibres = nfibres
            self.pred_cell_future = False
            
            self.reg_reg = cer_rate_reg
            self.reg_type = 'l1'
            
            self.bias = bias
            
            if num_hidden_layers > 0:
                self.update_output_only = opt_out_weight_only
            else:
                self.update_output_only = False
                        
            self.require_jacs = False
            
            self.errors = [] #keep track of TD errors
            
            if output_size is None: 
                self.output_size = input_size
            else:
                self.output_size = output_size

            if self.num_hidden_layers == 0:
                self.update_etraces = self.update_etraces_linear
            elif self.num_hidden_layers == 1:
                if self.update_output_only:
                    self.update_etraces = self.update_etraces_one_layer_output                    
                else:
                    self.update_etraces = self.update_etraces_one_layer
            elif self.num_hidden_layers == 2 and self.update_output_only:
                self.update_etraces = self.update_etraces_two_layers_output
            else:
                self.update_etraces = self.update_etraces_slow
                self.require_jacs = True
            
            self.init_weights()

            self.requires_grad_(requires_grad=False)
    
    def init_weights(self):
        top_layer_dim = self.output_size if self.num_hidden_layers == 0 else self.hidden_size

        self.input_trigger = torch.nn.Linear(
            in_features=self.input_size, out_features=top_layer_dim, bias=self.bias
        )

        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(
                in_features=self.hidden_size,
                out_features=(
                    self.hidden_size if layer_index < self.num_hidden_layers - 1 else self.output_size
                ),
                bias=self.bias
            )
            for layer_index in range(self.num_hidden_layers)
        ])
        
        
        zero_init = True
        
        if zero_init:
            # zero-initialize the last layer, as in original Jadeberg et al. 2017
            if self.num_hidden_layers > 0:
                init.zeros_(self.layers[-1].weight)
                if self.bias:
                    init.zeros_(self.layers[-1].bias)
            else:
                init.zeros_(self.input_trigger.weight)
                if self.bias:
                    init.zeros_(self.input_trigger.bias)
                
        if self.nfibres is not None:
            
            def generate_synth_mask(input_shape, nfibres):
                mask = torch.ones(input_shape)
                for i in range(input_shape[0]):
                    connections = np.random.choice(input_shape[1], nfibres, replace=False)
                    mask[i, connections] = 0
                mask = mask.type(torch.bool)        
                return mask   
                        
            input_shape = self.input_trigger.weight.shape
            mask = generate_synth_mask(input_shape, self.nfibres)
            with torch.no_grad():
                self.input_trigger.weight[mask] = 0
                
    def reset_etraces(self):
        for param in self.parameters():
            shape = (self.batch_size, self.output_size,) + param.shape
            setattr(param, 'etrace', param.new_zeros(size=shape))
        self.old_jacs = None
    
    def compute_synth_grad(self):
        if self.final_step:
            return torch.zeros(self.batch_size, self.output_size, device=self.input.device)
        
        synth_grad = self.input_trigger(self.input)
         
        
        for layer in self.layers:         
            synth_grad = layer(torch.relu(synth_grad))
        
        return synth_grad
    
    def store_info(self, input, input_grad, imm_loss, synth_grad):
        self.old_input = input
        self.old_input_grad = input_grad

        self.old_synth_grad = synth_grad
        self.old_imm_loss = imm_loss
        
               
        if self.require_jacs:
            self.old_jacs = self.get_jac()

    def get_jac(self):
        jac = torch.autograd.functional.jacobian(self.nn_dummy, tuple(self.parameters()))
        return jac
    
    def nn_dummy(self, *params):
        for i, param in enumerate(self.parameters()):
            param.copy_(params[i].clone())
        synth_grad = self.compute_synth_grad()
        return synth_grad
            
    def forward(self, input, imm_loss, input_grad, first_forward=False, 
                final_step=False, verbose=False):
        self.verbose = verbose
        self.final_step = final_step
        self.first_forward = first_forward
        self.input = input
              
        #update weights *before* computing synthetic gradient...
        if not first_forward and self.train:
            self.imm_loss = imm_loss
            self.input_grad = input_grad 
            self.update_weights()
            
        synth_grad = self.compute_synth_grad()
        self.store_info(input, input_grad, imm_loss, synth_grad)
        
        return synth_grad
    
    def update_TD_error(self):
        if self.imm_loss is not None:
            if self.pred_cell_future:
                first_term = self.imm_loss
            else:
                first_term = torch.bmm(self.imm_loss[:, None, :], self.input_grad).reshape(self.batch_size, -1) 
        else:
            first_term = 0
        
        pred_t = self.compute_synth_grad()
        
        if self.final_step:
            sec_term = 0
        else:
            sec_term = torch.bmm(pred_t[:, None, :], self.input_grad).reshape(self.batch_size, -1)  
        
        self.delta_error = first_term + self.gamma * sec_term - self.old_synth_grad
        self.errors.append(torch.norm(self.delta_error).item())
    
    def update_etraces_slow(self):
        all_jacs = self.old_jacs
        for i, param in enumerate(self.parameters()):
            decay = torch.bmm(self.old_input_grad, param.etrace.reshape(
                                self.batch_size, self.input_size, -1)) #input_size x input_size, input_size x #synapses
            jac = all_jacs[i] #batch size, output_size, weight size
            param.etrace = self.lmbda*decay.reshape(param.etrace.shape) + jac
    
    def update_etraces_linear(self):
        for name, param in self.named_parameters():        
            param.etrace = self.gamma * self.lmbda * torch.bmm(self.old_input_grad, param.etrace.reshape(
                                self.batch_size, self.output_size, -1)).reshape(param.etrace.shape)        
            if 'weight' in name:
                toadd = self.old_input.repeat(1, self.output_size).reshape(
                            self.batch_size, self.output_size, self.input_size)
            elif 'bias' in name:
                toadd = 1
            param.etrace[:,  torch.arange(self.output_size),
                            torch.arange(self.output_size)] += toadd
                
    def update_etraces_one_layer(self):        
        hidd_activity = self.input_trigger(self.old_input)
        hidd_activity = torch.relu(hidd_activity)
        for param in self.parameters():                
            param.etrace = self.gamma * self.lmbda * torch.bmm(self.old_input_grad, param.etrace.reshape(
                                self.batch_size, self.input_size, -1)).reshape(param.etrace.shape)

        
        #output bias...
        self.layers[0].bias.etrace[:,  torch.arange(self.output_size),
                            torch.arange(self.output_size)] += 1
                
        #output weight...
        toadd = hidd_activity.repeat(1, self.input_size).reshape(
                    self.batch_size, self.hidden_size, self.output_size).permute(0, 2, 1)
        
        self.layers[0].weight.etrace[:,  torch.arange(self.output_size),
                        torch.arange(self.output_size)] += toadd   
        
        #input bias
        outweight = self.layers[0].weight.clone()

        toadd = torch.cat(self.hidden_size*[self.old_input]).reshape(
                    self.hidden_size, self.batch_size, self.output_size).permute(1, 0, 2)

        toadd[hidd_activity <= 0] = 0
        toadd = toadd.permute(0, 2, 1)

        self.input_trigger.bias.etrace += toadd
        
        #input weight
        toadd = outweight[None, :, :, None] * self.old_input[:, None, None, :]
        toadd = toadd.permute(0, 2, 1, 3)
        toadd[hidd_activity <= 0] = 0
        self.input_trigger.weight.etrace += toadd.permute(0, 2, 1, 3)
        
    def update_etraces_one_layer_output(self):        
        hidd_activity = self.input_trigger(self.old_input)
        hidd_activity = torch.relu(hidd_activity)
        for name, param in self.named_parameters():                
            if 'input' in name:
                continue
            elif 'weight' in name:
                toadd = hidd_activity.repeat(1, self.input_size).reshape(
                    self.batch_size, self.input_size, self.hidden_size)

            elif 'bias' in name:
                toadd = 1
                
            param.etrace = self.gamma * self.lmbda * torch.bmm(self.old_input_grad, param.etrace.reshape(
                                self.batch_size, self.input_size, -1)).reshape(param.etrace.shape)
        
            param.etrace[:,  torch.arange(self.input_size),
                            torch.arange(self.input_size)] += toadd
            
    def update_etraces_two_layers_output(self):
        hidd_activity = torch.relu(self.input_trigger(self.old_input))
        hidd_activity = torch.relu(self.layers[0](hidd_activity))
        for name, param in self.named_parameters():                
            if 'input' in name or '0' in name:
                continue
            elif 'weight' in name:
                toadd = hidd_activity.repeat(1, self.input_size).reshape(
                    self.batch_size, self.input_size, self.hidden_size)

            elif 'bias' in name:
                toadd = 1
                
            param.etrace = self.gamma * self.lmbda * torch.bmm(self.old_input_grad, param.etrace.reshape(
                                self.batch_size, self.input_size, -1)).reshape(param.etrace.shape)
        
            param.etrace[:,  torch.arange(self.input_size),
                            torch.arange(self.input_size)] += toadd
        
    def relu_prime(self, mat):
        mat_prime = torch.zeros_like(mat)
        mat_prime[mat > 0] = 1
        return mat_prime
    
    def update_weights(self):
        self.update_TD_error()     
        self.update_etraces()
        
        for name, param in self.named_parameters():
            if self.update_output_only:
                if 'input' in name:
                    continue
            toadd = torch.bmm(self.delta_error[:, None, :], param.etrace.reshape(
                                self.batch_size, self.output_size, -1))
            
            toadd = toadd.reshape((self.batch_size, ) + param.shape)
            
            #finally take the batch mean of these gradients
            toadd = torch.mean(toadd, dim=0)
            
            norm_orig = torch.norm(toadd)
            if norm_orig > 1:
                toadd = toadd/norm_orig

            if self.reg_reg > 0 and not self.first_forward:
                if 'bias' in name:
                    inp = torch.ones(self.batch_size, param.shape[0], device=toadd.device)
                elif 'input' in name:
                    inp = self.old_input
                else:
                    inp = torch.relu(self.input_trigger(self.old_input))
                reg_term = self.get_rate_reg(inp, self.old_synth_grad,
                                             bias='bias' in name)
                toadd += self.reg_reg * reg_term
                             
            if self.alpha_learning:
                param += self.alpha * toadd
            else:                
                if param.grad is None:
                    param.grad = - toadd
                else:
                    param.grad += - toadd

    
    def update_batch_size(self, batch_size):
        self.batch_size = batch_size
        
    def get_rate_reg(self, inp, out, bias=False):        
        if bias:
            dL_dW = inp
        if not bias:
            out = out[:, :, None]
            dL_dW = inp[:, None, :].repeat(1, self.output_size, 1)
        
        if self.reg_type == 'l1':
            dL_dW = torch.sign(out) * dL_dW
        elif self.reg_type == 'l2':
            dL_dW = out * dL_dW
            
        dL_dW = torch.mean(dL_dW, dim=0)
            
        return dL_dW
        
