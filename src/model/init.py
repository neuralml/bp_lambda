import os
import yaml
import torch
import torch.nn as nn
import torch.nn.init as init
from functools import wraps

try:
    from .lstm_bp_lambda import lstm_bp_lambda
    from .rnn_bp_lambda import rnn_bp_lambda
except:
    from lstm_bp_lambda import lstm_bp_lambda
    from rnn_bp_lambda import rnn_bp_lambda

###############################################################################
# RNN WRAPPERS
###############################################################################

class RNNPredictor(nn.Module):
    def __init__(self, rnn, output_size, predict_last=True):
        super(RNNPredictor, self).__init__()
        self.rnn = rnn
        self.linear = nn.Linear(rnn.hidden_size, output_size)
        self.predict_last = predict_last

    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.linear.reset_parameters()

    @property
    def input_size(self):
        return self.rnn.input_size

    @property
    def output_size(self):
        return self.linear.weight.shape[0]

    @property
    def hidden_size(self):
        return self.rnn.hidden_size

    @property
    def n_layers(self):
        return self.rnn.num_layers

    def forward(self, input, hidden=None):
        output, hidden = self.rnn(input, hidden)

        if self.predict_last:
            pred = self.linear(output[:, -1, :])
        else:
            tsteps = input.size(1)
            pred = [self.linear(output[:, i, :]) for i in range(tsteps)]
            pred = torch.stack(pred, dim=1)
            
        return pred, hidden
    
    def wrap_predlast_train(self, func):
        @wraps(func)
        def wrapper(training=True):
            self.predict_last = training
            return func(training)
        return wrapper

###############################################################################
# Model Initialization
###############################################################################

def _weight_init_(module, init_fn_):
    try:
        init_fn_(module.weight.data)
    except AttributeError:
        for layer in module.all_weights:
            w, r = layer[:2]
            init_fn_(w)
            init_fn_(r)


def weight_init_(rnn, mode=None, **kwargs):
    if mode == 'xavier':
        _weight_init_(rnn, lambda w: init.xavier_uniform_(w, **kwargs))
    elif mode == 'orthogonal':
        _weight_init_(rnn, lambda w: init.orthogonal_(w, **kwargs))
    elif mode == 'kaiming':
        _weight_init_(rnn, lambda w: init.kaiming_uniform_(w, **kwargs))
    elif mode != None:
        raise ValueError(
                'Unrecognised weight initialisation method {}'.format(mode))


def init_rnn(rnn_type, hidden_size, input_size, n_layers, output_size,
             dropout=0.0, weight_init=None, predict_last=True,
             apply_sg=False, synth_params=None, backprop=False, bptt=None,
             classification=False, batch_size=-1, opt_each_trunc=False,
             evec_scale=None, forget_gate=None, seqlen=None,
             alpha=0., use_etraces=True):    
    if rnn_type == 'RNN':
        rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

    elif rnn_type == 'LSTM':
        assert not apply_sg, 'Chosen model is LSTM but apply_sg. Choose model LSTM_dni instead'
        rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

    elif rnn_type == 'GRU':
        rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )        
    elif rnn_type in ['rnn_bp_lambda', 'lstm_bp_lambda']:
        
        ind = ['rnn_bp_lambda', 'lstm_bp_lambda'].index(rnn_type)
        model_func = [rnn_bp_lambda, lstm_bp_lambda][ind]
        
        rnn = model_func(input_size=input_size, 
            hidden_size=hidden_size,
            output_size = output_size,
            predict_last=predict_last,
            apply_sg=apply_sg,
            synth_params=synth_params,
            backprop=backprop, 
            bptt=bptt,
            classification=classification,
            batch_size=batch_size,
            evec_scale=evec_scale,
            forget_gate=forget_gate,
            seqlen=seqlen,
            alpha=alpha,
            use_etraces=use_etraces)
    else:
        raise ValueError('Unrecognized RNN type')

    if rnn_type in ['rnn_bp_lambda', 'lstm_bp_lambda']:
        model = rnn
    else:
        weight_init_(rnn, weight_init)
    
        model = RNNPredictor(
            rnn=rnn,
            output_size = output_size,
            predict_last=predict_last
        ) 
        model.classification = classification
        model.task_start = None
    
    model.rnn_type = rnn_type
    
    if opt_each_trunc:
        assert bptt is not None or (not model.backprop), 'Optimising after each truncation but bptt is None!'
    
    model.opt_each_trunc = opt_each_trunc

    return model

###############################################################################
#  Load models
###############################################################################

def load_meta(path):
    with open(path, mode='r') as f:
        meta = yaml.safe_load(f)
    return meta

def _load_model(meta, model_file):
    meta = load_meta(meta)
    with open(model_file, mode='rb') as f:
        state_dict = torch.load(f)
        if 'model-state' in state_dict:
            state_dict = state_dict['model-state']
    m = init_rnn(device='cpu', **meta['model-params'])
    m.load_state_dict(state_dict)

    return m


def load_model(model_folder):
    meta = os.path.join(model_folder, 'meta.yaml')
    for file in os.listdir(model_folder):
        if file.endswith((".pt", ".pth")):
            file = os.path.join(model_folder, file)
            return _load_model(meta, file)
