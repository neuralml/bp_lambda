"""
Run the copy-repeat task
"""

import sys, os
import inspect
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import numpy as np
import torch
from ignite.engine import Events
from sacred import Experiment
from sacred.observers import FileStorageObserver

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

save_path = os.path.join(parentdir, 'results/copy-repeat')

# Load experiment ingredients
from ingredients.dataset import copy_repeat as dataset, load_copy_repeat as load_dataset
    
from ingredients.model import model, init_model
from ingredients.training import training, init_metrics, init_optimizer, \
                                 create_rnn_trainer, create_rnn_evaluator, \
                                 Tracer, ModelCheckpoint

import logging
logging.getLogger("ignite").setLevel(logging.WARNING)

# Add configs
training.add_config(currentdir + '/configs/training_copy_repeat.yaml')
dataset.add_config(currentdir + '/configs/dataset_copy_repeat.yaml')
model.add_config(currentdir + '/configs/model_copy_repeat.yaml')

ex_name = 'copy-repeat'
ex = Experiment(name=ex_name, ingredients=[dataset, model, training],)
ex.add_config(no_cuda=False, save_folder = os.path.join(save_path, 'temp'),
              task_name='copy-repeat') 

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
    print("Beginning experiment: config is {}".format(_config))
    no_cuda = _config['no_cuda']
    epochs = _config['training']['epochs']
    bptt = _config['training']['bptt']

    input_size = _config['dataset']['bit_dimension'] + 2
    seq_len =  5 #initial sequence length
    output_size = input_size - 1   
 
    predict_last = False
    
    print("Number of epochs is", epochs)
    log_interval = epochs // 20
    
    log_interval = 100 
    
    batch_size = _config['training']['batch_size']
    save_folder = _config['save_folder']

    # Init metrics
    loss, metrics = init_metrics('xent_multiple', ['xent_multiple', 'acc_binary'])
    classification = True

    device = set_seed_and_device(seed, no_cuda)

    training_set, validation_set, test_set = load_dataset(batch_size=batch_size)
    print("loaded dataset") 

    model = init_model(input_size=input_size, 
                        output_size=output_size,
                        bptt=bptt, classification=classification,
                        batch_size=batch_size, seqlen=seq_len,
                        predict_last=predict_last)
    if 'bp_lambda' in model.rnn_type:
        model.get_output_derivative = model.get_output_derivative_nll_raw
        
    model.teacher_inds = torch.arange(3, 5)
    model.task_start = 3
    if 'bp_lambda' in model.rnn_type and model.bptt is not None:
        start = max(model.bptt - model.task_start -1, 0)
        model.teacher_inds = torch.arange(start, model.seqlen, device=device)
        model.task_length = 5
    model = model.to(device=device)

    print("Initialised model")
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

    @trainer.on(Events.ITERATION_COMPLETED)
    def increment_copy_repeat(engine):
        loss_batch = tracer._batch_trace[-1]
        loss_batch /= np.log(2) #in bits
        
        if loss_batch < 0.15:
            increment_epoch = engine.state.epoch
            print("batch loss is less than <0.15 bits:", loss_batch)
            print("incrementing to next setting! (epoch ", increment_epoch)
            training_set.increment()
            print("N={}; R={}".format(training_set.N, training_set.R))
            ex.log_scalar('increment_epoch', increment_epoch)  

            increment_iteration = engine.state.iteration   
            ex.log_scalar('increment_iteration', increment_iteration)      

            new_task_start = training_set.N + 2
            new_task_length = (training_set.R + 1) * training_set.N + 3
            
            model.task_start = new_task_start
            model.teacher_inds = torch.arange(new_task_start, new_task_length, device=device)
            model.seqlen_whole = new_task_length

            if 'bp_lambda' in model.rnn_type and model.bptt is not None:
                start = max(model.bptt - model.task_start -1, 0)
                model.teacher_inds = torch.arange(start, model.bptt, device=device)  
                model.task_length = new_task_length

    print("Running training")
    # Run the training
    trainer.run(training_set, max_epochs=epochs)

    final_model_path = save_path + 'final-model.pt'
    torch.save(model.state_dict(), final_model_path)
    ex.add_artifact(final_model_path, 'final-model.pt')
    os.remove(final_model_path)
    
    print("Experiment finished!")
