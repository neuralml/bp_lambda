"""
Run the line-drawing (target-reaching) task
"""

import sys, os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import inspect
import numpy as np
import torch
from ignite.engine import Events
from sacred import Experiment
from sacred.observers import FileStorageObserver

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

save_path = os.path.join(parentdir, 'results/linedraw')

# Load experiment ingredients
from ingredients.dataset import linedraw as dataset, load_linedraw as load_dataset 
from ingredients.model import model, init_model
from ingredients.training import training, init_metrics, init_optimizer, \
                                 create_rnn_trainer, create_rnn_evaluator, \
                                 Tracer, ModelCheckpoint

import logging
logging.getLogger("ignite").setLevel(logging.WARNING)

# Add configs
training.add_config(currentdir + '/configs/training_linedraw.yaml')
dataset.add_config(currentdir + '/configs/dataset_linedraw.yaml')
model.add_config(currentdir + '/configs/model_linedraw.yaml')

ex_name = 'linedraw'
ex = Experiment(name=ex_name, ingredients=[dataset, model, training],)
ex.add_config(no_cuda=False, save_folder = os.path.join(save_path, 'temp'),
              task_name='linedraw') 

ex.add_package_dependency('torch', torch.__version__)

ex.observers.append(FileStorageObserver.create(save_path))

# Functions
@ex.capture
def set_seed_and_device(seed, no_cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and not no_cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device

@ex.automain
def main(_config, seed):
    print("Beginning experiment. Config is {}".format(_config))
    no_cuda = _config['no_cuda']
    epochs = _config['training']['epochs']
    bptt = _config['training']['bptt']

    input_size = _config['dataset']['input_D']
    output_size = 2
    seq_len =  _config['dataset']['seq_len']
    predict_last = _config['dataset']['end_only']
    
    log_interval = epochs // 20
    
    log_interval = 20
    
    batch_size = _config['training']['batch_size']

    loss, metrics = init_metrics('mse', ['mse'])
    classification = False

    device = set_seed_and_device(seed, no_cuda)

    training_set, test_set, validation_set = load_dataset(batch_size=batch_size)
   
    model = init_model(input_size=input_size, 
                        output_size=output_size,
                        bptt=bptt, classification=classification,
                        batch_size=batch_size, seqlen=seq_len,
                        predict_last=predict_last)
                               
    model = model.to(device=device)
    print("initialised model") 

    if _config['dataset']['end_only']:
        model.train = model.wrap_predlast_train(model.train)    
    
    optimizer = init_optimizer(model=model)

    # Init engines
    trainer = create_rnn_trainer(model, optimizer, loss, device=device)

    @trainer.on(Events.EPOCH_STARTED)
    def print_epoch(engine):
        nepoch = engine.state.epoch
        if nepoch % log_interval == 0:
            print('#'*75)
            print("Epoch: {}".format(engine.state.epoch))   
            print('#'*75)
    
    # Exception for early termination
    @trainer.on(Events.EXCEPTION_RAISED)
    def terminate(engine, exception):
        if isinstance(exception, KeyboardInterrupt):
            engine.should_terminate=True

    # Record training progression
    tracer = Tracer(metrics).attach(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training(engine):
        loss_epoch = tracer.loss[-1]
        if isinstance(loss, torch.nn.modules.loss.MSELoss):
            loss_epoch = loss_epoch * output_size
        ex.log_scalar('training_loss', loss_epoch)
        tracer.loss.clear()

    print("Running training")
    # Run the training
    trainer.run(training_set, max_epochs=epochs)

    final_model_path = save_path + 'final-model.pt'
    torch.save(model.state_dict(), final_model_path)
    ex.add_artifact(final_model_path, 'final-model.pt')
    os.remove(final_model_path)

    print("Experiment finished!")

