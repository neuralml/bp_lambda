import sys
from sacred import Ingredient
from ignite.engine import Events

from training.handlers import *
from training.loss import init_metrics
from training.optimizer import init_optimizer, init_lr_scheduler
from training.engine import create_rnn_trainer, create_rnn_evaluator

training = Ingredient('training',)

init_metrics = training.capture(init_metrics)
init_optimizer = training.capture(init_optimizer)
create_rnn_trainer = training.capture(create_rnn_trainer)
create_rnn_evaluator = training.capture(create_rnn_evaluator)
init_lr_scheduler = training.capture(init_lr_scheduler)
