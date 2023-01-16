import csv
import numpy as np
import torch
import ignite.handlers as hdlrs
from ignite.engine import Events
from tqdm import tqdm


class ModelCheckpoint(hdlrs.ModelCheckpoint):
    @property
    def best_model(self):
        with open(self.last_checkpoint, mode='rb') as f:      
            state_dict = torch.load(f)

        return state_dict


class Tracer(object):
    def __init__(self, val_metrics, save_path=None, save_interval=1, save_batch_loss=False):
        self.metrics = ['loss']
        self.loss = []
        self._batch_trace = []
        self.save_path = save_path
        self.save_interval = save_interval

        if save_batch_loss:
            self._batch_trace_all = []
        self.save_batch_loss= save_batch_loss

        template = 'val_{}'
        for k in val_metrics:
            name = template.format(k)
            setattr(self, name, [])
            self.metrics.append(name)

    def _initalize_traces(self, engine):
        for k in self.metrics:
            getattr(self, k).clear()

    def _save_batch_loss(self, engine):
        self._batch_trace.append(engine.state.output)

        if self.save_batch_loss:
            self._batch_trace_all.append(engine.state.output)


    def _trace_training_loss(self, engine):
        avg_loss = np.mean(self._batch_trace)
        self.loss.append(avg_loss)
        self._batch_trace.clear()

    def _trace_validation(self, engine):
        metrics = engine.state.metrics
        template = 'val_{}'
        for k, v in metrics.items():
            trace = getattr(self, template.format(k))
            trace.append(v)

    def attach(self, trainer, evaluator=None):
        trainer.add_event_handler(Events.STARTED, self._initalize_traces)
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, self._save_batch_loss)
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, self._trace_training_loss)

        if evaluator is not None:
            evaluator.add_event_handler(
                Events.COMPLETED, self._trace_validation)

        if self.save_path is not None:
            trainer.add_event_handler(
                Events.EPOCH_COMPLETED, self._save_at_interval)

        return self

    def _save_at_interval(self, engine):
        if engine.state.iteration % self.save_interval == 0:
            self.save_traces()

    def save_traces(self):
        for loss in self.metrics:
            trace = getattr(self, loss)
            with open('{}/{}.csv'.format(self.save_path, loss), mode='w') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                for i, v in enumerate(trace):
                    wr.writerow([i + 1, v])


class Logger(object):
    def __init__(self, loader, log_interval, pbar=None, desc=None):
        n_batches = len(loader)
        self.desc = 'iteration-loss: {:.5f}' if desc is None else desc
        self.pbar = pbar or tqdm(
            initial=0, leave=False, total=n_batches,
            desc=self.desc.format(0)
        )
        self.log_interval = log_interval
        self.running_loss = 0
        self.n_batches = n_batches

    def _log_batch(self, engine):
        self.running_loss += engine.state.output

        iter = (engine.state.iteration - 1) % self.n_batches + 1
        if iter % self.log_interval == 0:
            self.pbar.desc = self.desc.format(
                engine.state.output)
            self.pbar.update(self.log_interval)

    def _log_epoch(self, engine):
        self.pbar.refresh()
        tqdm.write("Epoch: {} - avg loss: {:.5f}"
            .format(engine.state.epoch, self.running_loss / self.n_batches))
        self.running_loss = 0
        self.pbar.n = self.pbar.last_print_n = 0

    def _log_validation(self, engine):
        metrics = self.evaluator.state.metrics

        message = []
        for k, v in metrics.items():
            message.append("{}: {:.5f}".format(k, v))
        tqdm.write('\tvalidation: ' + ' - '.join(message))

    def attach(self, trainer, evaluator=None, metrics=None):
        trainer.add_event_handler(Events.ITERATION_COMPLETED, self._log_batch)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self._log_epoch)
        trainer.add_event_handler(Events.COMPLETED, lambda x: self.pbar.close())

        if evaluator is not None and metrics is None:
            raise ValueError('')

        if evaluator is not None:
            self.evaluator = evaluator
            trainer.add_event_handler(Events.EPOCH_COMPLETED, self._log_validation)

        return self
    
class ProgressLog(object):
    def __init__(self, n_batches, log_interval, pbar=None, desc=None):
        self.desc = 'iteration-loss: {:.5f}' if desc is None else desc
        self.pbar = pbar or tqdm(
            initial=0, leave=False, total=n_batches,
            desc=self.desc.format(0)
        )
        self.log_interval = log_interval
        self.running_loss = 0
        self.n_instances = 0
        self.n_batches = n_batches

    def attach(self, trainer, evaluator=None, metrics=None):
        def _log_batch(engine):
            batch_size = engine.state.batch[0].size(0)
            self.running_loss += engine.state.output * batch_size
            self.n_instances += batch_size

            iter = engine.state.iteration % self.n_batches
            if iter % self.log_interval == 0:
                self.pbar.desc = self.desc.format(
                    engine.state.output)
                self.pbar.update(self.log_interval)

        def _log_epoch(engine):
            self.pbar.refresh()
            tqdm.write("Epoch: {} - avg loss: {:.5f}"
                .format(engine.state.epoch, self.running_loss / self.n_instances))
            self.running_loss = 0
            self.n_instances = 0
            self.pbar.n = self.pbar.last_print_n = 0

        trainer.add_event_handler(Events.ITERATION_COMPLETED, _log_batch)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, _log_epoch)
        trainer.add_event_handler(Events.COMPLETED, lambda x: self.pbar.close())

        if evaluator is not None and metrics is None:
            raise ValueError('')

        if evaluator is not None:
            self.evaluator = evaluator
            def _log_validation(engine):
                metrics = self.evaluator.state.metrics

                message = []
                for k, v in metrics.items():
                    message.append("{}: {:.5f}".format(k, v))
                tqdm.write('\tvalidation: ' + ' - '.join(message))

            trainer.add_event_handler(Events.EPOCH_COMPLETED, _log_validation)

        return self


Timer = hdlrs.Timer
