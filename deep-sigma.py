"""
    [ACC=0.9875]    [NMI=0.9644]    [ARI=0.9723]
"""
from dataloader import get_mnist_all
import argparse
from tqdm import tqdm
from utils import *
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_,constant_
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()

    def forward(self, x):
        return x.view(x.size(0),-1)

def weight_inits(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        kaiming_normal_(m.weight.data,mode='fan_out',nonlinearity='relu')
        # m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        m.weight.data=torch.eye(10).cuda()
        m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        constant_(m.weight.data, 1)
        constant_(m.bias.data, 0)

class mnistNetwork(nn.Module):
    def __init__(self):
        super(mnistNetwork,self).__init__()

        self.backbone=nn.Sequential(
            #b*1*28*28
            nn.Conv2d(1,64,3,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #b*64*26*26
            nn.Conv2d(64, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #b*64*24*24
            nn.Conv2d(64, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            # b*64*11*11
            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #b*128*9*9
            nn.Conv2d(128, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # b*128*7*7
            nn.Conv2d(128, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            # b*128*2*2
            nn.Conv2d(128,10,1,1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(10),
            #b*10*1*1
            Flatten(),
            nn.Linear(10,10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

        self.backbone.apply(weight_inits)

    def forward(self, x):
        return self.backbone(x)

def sample_Z(batch, z_dim, sampler='one_hot', num_class=10, n_cat=1, label_index=None, nc = 0.1):
    if sampler == 'mul_cat':
        if label_index is None:
            label_index = np.random.randint(low= 0 , high = num_class, size = batch)
        return np.hstack((0.20 * np.random.randn(batch, z_dim-num_class*n_cat),
                          np.tile(np.eye(num_class)[label_index], (1, n_cat))))
    elif sampler == 'one_hot':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return np.hstack((nc * np.random.randn(batch, z_dim-num_class), np.eye(num_class)[label_index]))

    elif sampler == 'Identity':
        return np.hstack((nc * np.random.randn(batch, z_dim-num_class), np.eye(num_class)))

    elif sampler == 'one_hot_label':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return np.hstack((nc * np.random.randn(batch, z_dim-num_class), np.eye(num_class)[label_index])), label_index


parse = argparse.ArgumentParser('e')
parse.add_argument('--batch_size', type=int, default=128)
parse.add_argument('--niter', type=int, default=300)
parse.add_argument('--threshold', type=int, default=5000)

args = parse.parse_args()

device = torch.device('cuda')

encoder = mnistNetwork()
encoder.load_state_dict(torch.load('./CLU_model_148.pth'))
encoder.to(device)
encoder.eval()


if __name__ == '__main__':
    ##EVAL
    dataset = get_mnist_all(args)
    pre_y = []
    tru_y = []
    for data, label in dataset:
        data = data * 2.0 - 1.0

        f = encoder(data.view(-1, 1, 28, 28).cuda())
        pre_y.append(torch.argmax(f, 1).detach().cpu().numpy())
        tru_y.append(label.numpy())
    pre_y = np.concatenate(pre_y, 0)
    tru_y = np.concatenate(tru_y, 0)
    acc = ACC(tru_y, pre_y)
    nmi = NMI(tru_y, pre_y)
    ari = ARI(tru_y, pre_y)
    print('[ACC={:.4f}]\t[NMI={:.4f}]\t[ARI={:.4f}]'.format(acc, nmi, ari))
