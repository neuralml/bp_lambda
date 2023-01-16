#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training/validation loops defined using the engine library
"""

import sys, os
import inspect
import torch
import torch.nn as nn

from ignite.engine import Events, Engine, _prepare_batch

try:
    import dni
except:
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    if parentdir not in sys.path:
        sys.path.insert(0, parentdir) 
    import dni

########################################################################################
# Training
########################################################################################

def _detach_hidden_state(hidden_state):
    """
    Use this method to detach the hidden state from the previous batch's history.
    This way we can carry hidden states values across training which improves
    convergence  while avoiding multiple initializations and autograd computations
    all the way back to the start of start of training.
    """

    if hidden_state is None:
        return None
    elif isinstance(hidden_state, torch.Tensor):
        return hidden_state.detach()
    elif isinstance(hidden_state, list):
        return [_detach_hidden_state(h) for h in hidden_state]
    elif isinstance(hidden_state, tuple):
        return tuple(_detach_hidden_state(h) for h in hidden_state)
    raise ValueError('Unrecognized hidden state type {}'.format(type(hidden_state)))


def create_rnn_trainer(model, optimizer, loss_fn, grad_clip=0, reset_hidden=True,
                    device=None, non_blocking=False, prepare_batch=_prepare_batch,
                    bptt=None,):
    if device:
        model.to(device)
             
    if model.opt_each_trunc:
        model.optimizer = optimizer
               
    def _training_loop(engine, batch):
        # Set model to training and zero the gradients
        model.train()
        optimizer.zero_grad()
        
        # Load the batches
        inputs, targets = prepare_batch(batch, device=device, non_blocking=non_blocking)
        hidden = engine.state.hidden
  
        if model.rnn_type in ['LSTM', 'RNN']: #pytorch RNN/LSTM models used as controls
            pred, hidden = model(inputs, hidden)
            
            if model.task_start is not None:
                pred = pred[:,model.task_start:]
                targets = targets[:, model.task_start:] 

            loss = loss_fn(pred, targets)
            loss.backward()
        elif model.rnn_type in ['rnn_bp_lambda', 'lstm_bp_lambda']: #custom RNN/LSTM models used for experiments
            pred, hidden = model(inputs, target=targets, hidden=hidden)
            
            if model.task_start is not None:
                pred = pred[:,model.task_start:]
                targets = targets[:, model.task_start:] 
                            
            if model.classification and not model.predict_last and pred.shape!= targets.shape:
                pred = pred.permute(0, 2, 1)

            loss = loss_fn(pred, targets)
            
            if model.backprop:
                if model.classification and not model.predict_last and False: #penn dataset
                    loss_for_backward = loss * targets.shape[1] #correct for averaging vs backprop free eprop
                    loss_for_backward.backward()
                else:
                    loss.backward()

        if not model.opt_each_trunc:
            optimizer.step()

        if not reset_hidden:
            engine.state.hidden = hidden

        return loss.item()
    
    def _training_loop_truncs(engine, batch):
        # Set model to training and zero the gradients
        model.train()
        optimizer.zero_grad()
        # Load the batches
        inputs, targets = prepare_batch(batch, device=device, non_blocking=non_blocking)
            
        seq_len = inputs.shape[1]
        ntrunc = (seq_len - 1) // bptt + 1 
        
        sparse_targets = False
        if model.task_start is not None: #copy-repeat task
            targets = targets[:, model.task_start:] 
            sparse_targets = True
            window_start = seq_len - targets.shape[1]
                  
        
        total_loss = 0 #keep tally of total loss for the batch sequence
        hidden = engine.state.hidden
        
        targets_trunc = None
        sparse_seqlen = 0

        remainder = seq_len % bptt #the remainder 
        if remainder == 0:
            remainder = bptt

        for i in range(0, ntrunc):

            if i == 0:
                trunc_start = 0
                trunc_end = remainder
            else:
                trunc_start = remainder + (i-1) * bptt
                trunc_end = trunc_start + bptt

            inputs_trunc = inputs[:,trunc_start:trunc_end]
            
            if not model.predict_last:
                if model.task_start is not None: #copy-repeat
                    end = trunc_end - window_start
                    if trunc_start <= window_start and trunc_end > window_start:
                        end = trunc_end - window_start
                        targets_trunc = targets[:, :end]
                    elif trunc_start > window_start:
                        start = trunc_start - window_start
                        targets_trunc = targets[:, start:end]   
                    else:
                        targets_trunc = None 
                else:
                    targets_trunc = targets[:,i*bptt:(i+1)*bptt]
            else:
                if i == (ntrunc - 1):
                    targets_trunc = targets
                
            backward_loss = (not model.predict_last) or (i == ntrunc - 1)
            
            # Starting each truncation, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the batch.
            #print("About to detach hidden")
            detached_hidden = _detach_hidden_state(hidden)
            
            if model.rnn_type in ['RNN', 'LSTM']:
                pred, new_hidden = model(inputs_trunc, detached_hidden)      
                
                if targets_trunc is not None and targets_trunc.shape[1] > 0:                        
                    if backward_loss:
                        loss = loss_fn(pred, targets_trunc)
                        loss.backward()
            elif model.rnn_type in ['rnn_bp_lambda', 'lstm_bp_lambda']:
                if model.backprop and model.apply_sg: #n-step DNI case
                    with dni.defer_backward():
                        pred, new_hidden = model(inputs_trunc, hidden=detached_hidden, 
                                             target=targets_trunc, latter_trunc=(i>0)) 
                            
                        if targets_trunc is not None and (targets_trunc.ndim == 1 or targets_trunc.shape[1] > 0):
                            if sparse_targets:
                                if targets_trunc.shape[1] < pred.shape[1]:
                                    pred = pred[:, pred.shape[1] - targets_trunc.shape[1]:] 
                                sparse_seqlen += pred.shape[1]

                                if pred.shape!= targets_trunc.shape:
                                    pred = pred.permute(0, 2, 1)
                            if backward_loss:
                                loss = loss_fn(pred, targets_trunc)
                                dni.backward(loss)
                        else:
                            backward_loss = False
                else:
                    pred, new_hidden = model(inputs_trunc, hidden=detached_hidden, 
                                         target=targets_trunc, latter_trunc=(i>0))   
                                     
                    if backward_loss:
                        if targets_trunc is not None and (targets_trunc.ndim == 1 or targets_trunc.shape[1] > 0):    
                            if sparse_targets:
                                if targets_trunc.shape[1] < pred.shape[1]:
                                    pred = pred[:, pred.shape[1] - targets_trunc.shape[1]:]
                                sparse_seqlen += pred.shape[1]
                                if pred.shape!= targets_trunc.shape:
                                    pred = pred.permute(0, 2, 1)
                                
                            loss = loss_fn(pred, targets_trunc)
                        
                            if model.backprop:
                                loss.backward()
                        else:
                            backward_loss = False
                                        
            if backward_loss:
                ntsps = 1 if targets_trunc.ndim == 1 else targets_trunc.shape[1]
                total_loss += ntsps*loss.item()
            
            if model.opt_each_trunc:
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)     
                
                optimizer.step()                    
                optimizer.zero_grad()

            hidden = new_hidden

        if not model.opt_each_trunc:

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)      
                           
            optimizer.step()

        if model.predict_last:
            reported_loss = loss.item()
        else:
            if sparse_targets: #normalise loss
                reported_loss = total_loss/sparse_seqlen
            else:
                reported_loss = total_loss/seq_len

        if not reset_hidden:
            engine.state.hidden = hidden
        return reported_loss

    # If reusing hidden states, detach them from the computation graph
    # of the previous batch. Usin the previous value may speed up training
    # but detaching is needed to avoid backprogating to the start of training.
    def _detach_wrapper(engine):
        if not reset_hidden:
            engine.state.hidden = _detach_hidden_state(engine.state.hidden)
                
    loop = _training_loop if bptt is None else _training_loop_truncs
    
    engine = Engine(loop)
    engine.add_event_handler(Events.EPOCH_STARTED, lambda e: setattr(e.state, 'hidden', None))
    engine.add_event_handler(Events.ITERATION_STARTED, _detach_wrapper)

        
    return engine


def create_rnn_evaluator(model, metrics, device=None, hidden=None, non_blocking=False,
                        prepare_batch=_prepare_batch, bptt=None, noisy=False):
    if noisy:
        assert bptt is not None, 'Noise is set to true but bptt is set to None'
    if device:
        model.to(device)
        
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs, targets = prepare_batch(batch, device=device, non_blocking=non_blocking)
            if model.rnn_type != 'LSTM_eprop':
                pred, _ = model(inputs, hidden)
            else:
                pred, _ = model(inputs)
            
            if model.classification and not model.predict_last and pred.shape!= targets.shape:
                pred = pred.permute(0, 2, 1)     
            return pred, targets    

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def run_training(
        model, train_data, trainer, epochs,
        metrics, test_data, model_checkpoint, device
    ):
    trainer.run(train_data, max_epochs=epochs)
    
    # Select best model
    best_model_path = model_checkpoint.last_checkpoint
    with open(best_model_path, mode='rb') as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)

    tester = create_rnn_evaluator(model, metrics, device=device)
    tester.run(test_data)
    
    return tester.state.metrics
