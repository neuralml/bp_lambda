import torch
import torch.nn as nn
import torch.nn.modules.loss as L
import torch.nn.functional as F
import ignite.metrics as M

def logit_transform(x):
    ypred, y = x
    ypred = torch.round(torch.sigmoid(ypred))
    return ypred, y

class RNNLossWrapper(L._Loss):
    def __init__(self, loss_fn, reduction='mean'):
        super(RNNLossWrapper, self).__init__(reduction=reduction)
        self._loss_fn = loss_fn

    def forward(self, input, target):
        pred, _ = input
        return self._loss_fn(pred, target)

class RateDecay(L._Loss):
    def __init__(self, target=0, reduction='mean', device='cpu'):
        self.target = torch.as_tensor(target).to(device)

    def forward(self, input, target):
        _, hidden = input

        return F.l1_loss(hidden, self.target, reduction=self.reduction)


class L1WeightDecay(L._Loss):
    def __init__(self, model_parameters, target=0, device='cpu'):
        super(L1WeightDecay, self).__init__('none')
        self.parameters = model_parameters
        self.target = torch.as_tensor(target).to(device)

    def forward(self, input, target):
        return F.l1_loss(self.parameters, self.target, reduction='none')


class ComposedLoss(L._Loss):
    def __init__(self, terms, weights):
        super(ComposedLoss, self).__init__(reduction='none')

        self.terms = terms
        self.weights = weights

    def forward(self, input, target):
        value = 0
        for term, w in zip(self.terms, self.weights):
            value += w * term(input, target)

        return value


def get_loss_fn(loss_fn_name):
    if loss_fn_name == 'mse':
        return nn.MSELoss()
    elif loss_fn_name == 'xent':
        return nn.CrossEntropyLoss()
    elif loss_fn_name == 'xent_multiple':
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError('Unknown loss function {}'.format(loss_fn_name))


def get_metric(metric):
    if metric == 'mse':
        return M.MeanSquaredError()
    elif metric == 'xent':
        return M.Loss(nn.CrossEntropyLoss())
    elif metric == 'xent_multiple':
        return M.Loss(torch.nn.BCEWithLogitsLoss())
    elif metric == 'acc':
        return M.Accuracy()
    elif metric == 'acc_binary':
        return M.Accuracy(output_transform=logit_transform)
    raise ValueError('Unrecognized metric {}.'.format(metric))


def init_metrics(loss, metrics, rate_reg=0.0, rnn_eval=True):
    criterion = get_loss_fn(loss)

    if rate_reg > 0:
        criterion = ComposedLoss(
            terms=[criterion, RateDecay()],
            decays=[1.0, rate_reg]
        )

    metrics = {m: get_metric(m) for m in metrics}

    return criterion, metrics
