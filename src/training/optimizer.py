import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def init_optimizer_old(optimizer, params, lr=0.01, l2_norm=0.0, **kwargs):

    if optimizer == 'adam':
        optimizer = optim.Adam(params,
            lr=lr, eps=1e-9, weight_decay=l2_norm, betas=[0.9, 0.98])
    elif optimizer == 'sparseadam':
        optimizer = optim.SparseAdam(params,
            lr=lr, eps=1e-9, weight_decay=l2_norm, betas=[0.9, 0.98])
    elif optimizer == 'adamax':
        optimizer = optim.Adamax(params,
            lr=lr, eps=1e-9, weight_decay=l2_norm, betas=[0.9, 0.98])
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(params,
            lr=lr, eps=1e-10, weight_decay=l2_norm, momentum=0.9)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(params,
            lr=lr, weight_decay=l2_norm)
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(params,
            lr=lr, weight_decay=l2_norm, lr_decay=0.9)
    elif optimizer == 'adadelta':
        optimizer = optim.Adadelta(params,
            lr=lr, weight_decay=l2_norm, rho=0.9)
    else:
        raise ValueError(r'Optimizer {0} not recognized'.format(optimizer))

    return optimizer

def init_optimizer(optimizer, model, lr_rnn=0.001, lr_readout=0.001, lr_synthesiser=0.001, l2_norm=0.0, **kwargs):
    #use rnn as base 
    if model.rnn_type in ['RNN', 'LSTM']:
        rnn_list = ['rnn.weight_hh_l0', 'rnn.weight_ih_l0', 'rnn.bias_hh_l0', 'rnn.bias_ih_l0']
        readout_list = ['linear.weight', 'linear.bias']
    elif model.rnn_type == 'LSTM_dni':
        rnn_list = ['rnn.rnn.weight_hh_l0', 'rnn.rnn.weight_ih_l0', 'rnn.rnn.bias_hh_l0', 'rnn.rnn.bias_ih_l0']
        readout_list = ['linear.weight', 'linear.bias']        
    else:
        rnn_list = ['w_ih', 'w_hh', 'b']
        readout_list = ['w_out', 'b_out']
    
    rnn_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in rnn_list, model.named_parameters()))))
    readout_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in readout_list, model.named_parameters()))))
    
    l2_norm_rnn = l2_norm
    
    l2_norm = 0
    params = [{'params': rnn_params, 'weight_decay': l2_norm_rnn, 'lr': lr_rnn},
                  {'params': readout_params, 'lr': lr_readout, 'weight_decay': l2_norm},]
    
    if hasattr(model, 'synthesiser'):
        params.append({'params': model.synthesiser.parameters(), 'lr': lr_synthesiser, 'weight_decay': l2_norm
                       })
    elif hasattr(model, 'backward_interface'):
        params.append({'params': model.backward_interface.parameters(), 'lr': lr_synthesiser, 'weight_decay': l2_norm
                       })  
    elif hasattr(model, 'rnn') and hasattr(model.rnn, 'backward_interface'):
        params.append({'params': model.rnn.backward_interface.parameters(), 'lr': lr_synthesiser, 'weight_decay': l2_norm
                       })        
        
    if hasattr(model, 'auto_encoder'):
        params.append({'params': model.auto_encoder.parameters()})

    if optimizer == 'adam':
        optimizer = optim.Adam(params, eps=1e-9, betas=[0.9, 0.98])
    elif optimizer == 'sparseadam':
        optimizer = optim.SparseAdam(params,
            lr=lr_rnn, eps=1e-9, weight_decay=l2_norm, betas=[0.9, 0.98])
    elif optimizer == 'adamax':
        optimizer = optim.Adamax(params,
            lr=lr_rnn, eps=1e-9, weight_decay=l2_norm, betas=[0.9, 0.98])
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(params,
            lr=lr_rnn, eps=1e-10, weight_decay=l2_norm, momentum=0.9)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(params,
            lr=lr_rnn, weight_decay=l2_norm, momentum=0.9) # 0.01
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(params,
            lr=lr_rnn, weight_decay=l2_norm, lr_decay=0.9)
    elif optimizer == 'adadelta':
        optimizer = optim.Adadelta(params,
            lr=lr_rnn, weight_decay=l2_norm, rho=0.9)
    else:
        raise ValueError(r'Optimizer {0} not recognized'.format(optimizer))

    return optimizer


def init_lr_scheduler(optimizer, scheduler, lr_decay, patience, threshold=1e-4, min_lr=1e-9):

    if scheduler == 'reduce-on-plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=lr_decay,
            patience=patience,
            threshold=threshold,
            min_lr=min_lr
        )
    else:
        raise ValueError(r'Scheduler {0} not recognized'.format(scheduler))

    return scheduler