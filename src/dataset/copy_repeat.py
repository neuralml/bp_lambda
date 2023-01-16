#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copy-repeat task as in 'Neural Turing Machines', Graves et al. 2014
Given pattern of length N and character R, model must repeat pattern R times
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class BatchGenerator:
    def __init__(
        self, size=1000, batch_size=100, bit_dimension=8, 
        offline_data=False, random_state=None, norm_repeat_val=None
    ):
        self.size = size
        self.batch_size = batch_size
        self.bit_dimension = bit_dimension
        self.offline_data = offline_data
        self.norm_repeat_val = norm_repeat_val

        if random_state is None or isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)

        self.rng = random_state
        self.init_state = random_state.get_state()
        
        self.N = 1 #length of sequence
        self.R = 1 #number of repeats

    def __iter__(self):
        for _ in range(len(self)):
            yield self.next_batch()

        if self.offline_data:
            self.reset()

    def __len__(self):
        return int(np.ceil(self.size/self.batch_size))

    def n_instances(self):
        return self.size * self.batch_size

    def reset(self):
        self.rng.set_state(self.init_state)
        
    def increment(self):
        if self.R < self.N:
            self.R += 1
        else:
            self.N += 1

    def get_norm_repeat(self):
        if self.norm_repeat_val is not None:
            return self.R/self.norm_repeat_val
        else:
            return self.R

    def sample_data(self,
    ):
        total_seq_len = (self.R + 1) * self.N + 3
        
        inputs_all = np.zeros((self.batch_size, total_seq_len, self.bit_dimension+2)) #delimieter at start + nrepeat
        
        targets_all = np.zeros((self.batch_size, total_seq_len, self.bit_dimension+1)) #delimiter at end
        
        sequence = self.rng.randint(0, 2, (self.batch_size, self.N, self.bit_dimension))   
        sequence_repeated = np.tile(sequence, (1, self.R,1))
        
        inputs_all[:, 0, self.bit_dimension] = 1 #delimter at start
        inputs_all[:, 1:self.N+1, :self.bit_dimension] = sequence
        inputs_all[:, self.N+1, self.bit_dimension+1] = self.get_norm_repeat()
        
        targets_all[:, self.N+2:-1, :self.bit_dimension] = sequence_repeated
        targets_all[:, -1, self.bit_dimension] = 1
           
        return inputs_all, targets_all

    def next_batch(self):
        inputs, targets = self.sample_data()

        inputs = torch.as_tensor(inputs, dtype=torch.float)
        targets = torch.as_tensor(targets, dtype=torch.float)

        return inputs, targets

    def torch_dataset(self):
        current_state = self.rng.get_state()
        self.rng.set_state(self.init_state)

        inputs, targets = zip(*[batch for batch in self])

        self.rng.set_state(current_state)

        data = TensorDataset(torch.cat(inputs), torch.cat(targets))

        return DataLoader(
            dataset=data,
            batch_size=self.batch_size, shuffle=True
        )


def load_copy_repeat(
    training_size=1000,
    test_size=1000,
    batch_size=100,
    train_val_split=0.2,
    fixdata=False,
    random_state=None,
    bit_dimension=8,
    norm_repeat_val=None,
):
    
    N = int(training_size * (1 - train_val_split))
    val_size = training_size - N
    

    if random_state is None:
        train_rng = np.random.randint(2**16-1)
        val_rng = np.random.randint(2**16-1)
        test_rng = np.random.randint(2**16-1)
    else:
        train_rng = random_state.randint(2**16-1)
        val_rng = random_state.randint(2**16-1)
        test_rng = random_state.randint(2**16-1)

    training_data = BatchGenerator(
        size=training_size, batch_size=batch_size, offline_data=fixdata, 
        random_state=train_rng, bit_dimension=bit_dimension,
        norm_repeat_val=norm_repeat_val)
    
    
    validation_data = BatchGenerator(
        size=training_size, batch_size=val_size, offline_data=fixdata, 
        random_state=val_rng, bit_dimension=bit_dimension,
        norm_repeat_val=norm_repeat_val)
    
    test_data = BatchGenerator(
        size=training_size, batch_size=test_size, offline_data=fixdata, 
        random_state=test_rng, bit_dimension=bit_dimension,
        norm_repeat_val=norm_repeat_val)


    if fixdata:
        training_data  = training_data.torch_dataset()
        test_data = test_data.torch_dataset()
        val_size = validation_data.torch_dataset()

    return training_data, validation_data, test_data

