import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as trans
import torchvision.datasets as dataset

import numpy as np

def load_raw(data_path, input_size, download=False, transform=None, target_transform=None):
    transform = transform or trans.Compose([
        trans.ToTensor(),
        trans.Lambda(lambda x: x.view(-1, input_size))
    ])

    train_data = dataset.MNIST(
        root=data_path, train=True, transform=transform, target_transform=target_transform, download=download)
    test_data = dataset.MNIST(
        root=data_path, train=False, transform=transform, target_transform=target_transform, download=download)

    return train_data, test_data


def load_mnist(data_path, input_size, batch_size, val_split=0.2,
        shuffle=True, download=False, transform=None, target_transform=None, numbers=None
    ):
    train_raw, test_raw = load_raw(data_path, input_size, download=download,
                                   transform=transform, target_transform=target_transform)
    
    if numbers is not None:
        train_raw = filter_mnist(train_raw, numbers)
        test_raw = filter_mnist(test_raw, numbers)
    
    # Split train data into training and validation sets
    N = len(train_raw)
    val_size = int(N * val_split)
    train_raw, validation_raw = random_split(
        train_raw, [N - val_size, val_size])
    
    train_data = DataLoader(
        train_raw, batch_size=batch_size, shuffle=shuffle)
    validation_data = DataLoader(
        validation_raw, batch_size=batch_size, shuffle=False)
    test_data = DataLoader(
        test_raw, batch_size=batch_size, shuffle=False)

    return train_data, validation_data, test_data


def filter_mnist(mnist_dataset, numbers):
    inds = np.isin(mnist_dataset.targets, numbers)
    
    mnist_dataset.data = mnist_dataset.data[inds]
    mnist_dataset.targets = mnist_dataset.targets[inds]
    
    return mnist_dataset





