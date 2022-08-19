from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader,TensorDataset
import torch
from torchvision import transforms
import numpy as np

def get_mnist_all(args):
    train_data = MNIST(root='../data', train=True, download=True)
    test_data = MNIST(root='../data', train=False, download=True)


    x_train = train_data.data.float()
    x_train = x_train.view(x_train.shape[0], 28, 28, 1) / 255.
    y_train = train_data.targets

    x_test = test_data.data.float()
    x_test = x_test.view(x_test.shape[0], 28, 28, 1) / 255.
    y_test = test_data.targets


    ds = TensorDataset(
        torch.cat([x_train, x_test], 0),
        torch.cat([y_train, y_test], 0)
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    return dl
