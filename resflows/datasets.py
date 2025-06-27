import torch
import torchvision.datasets as vdsets


class Dataset(object):

    def __init__(self, loc, transform=None, in_mem=True):
        self.in_mem = in_mem
        self.dataset = torch.load(loc)
        if in_mem: self.dataset = self.dataset.float().div(255)
        self.transform = transform

    def __len__(self):
        return self.dataset.size(0)

    @property
    def ndim(self):
        return self.dataset.size(1)

    def __getitem__(self, index):
        x = self.dataset[index]
        if not self.in_mem: x = x.float().div(255)
        x = self.transform(x) if self.transform is not None else x
        return x, 0


class MNIST(object):

    def __init__(self, dataroot, train=True, transform=None):
        self.mnist = vdsets.MNIST(dataroot, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.mnist)

    @property
    def ndim(self):
        return 1

    def __getitem__(self, index):
        return self.mnist[index]


class CIFAR10(object):

    def __init__(self, dataroot, train=True, transform=None):
        self.cifar10 = vdsets.CIFAR10(dataroot, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.cifar10)

    @property
    def ndim(self):
        return 3

    def __getitem__(self, index):
        return self.cifar10[index]

