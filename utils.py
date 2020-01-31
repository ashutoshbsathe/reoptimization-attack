import sys 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 
from torch.utils.data.sampler import SubsetRandomSampler as SRS 
import torch.utils.data as data_utils 
import numpy as np 

MNIST_DATA_ROOT = './mnist_data/'
CIFAR10_DATA_ROOT = './cifar10_data/'
FMNIST_DATA_ROOT = './fmnist_data/'

SEED = 161803398 # Golden Ratio

def get_mnist_data_loaders(batch_size=64, n_train=50000, \
    n_val=10000, n_test=10000, train_transform=None, \
    val_transform=None, test_transform=None):

    assert n_train + n_val == 60000
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    train_set = datasets.MNIST(root=MNIST_DATA_ROOT, download=True, \
        train=True, transform=train_transform)
    val_set = datasets.MNIST(root=MNIST_DATA_ROOT, download=True, \
        train=True, transform=val_transform)
    test_set = datasets.MNIST(root=MNIST_DATA_ROOT, download=True, \
        train=False, transform=test_transform)

    indices = np.arange(0, 60000)
    np.random.seed(SEED)
    np.random.shuffle(indices)

    train_sampler = SRS(indices[:n_train])
    val_sampler = SRS(indices[n_train:])
    test_sampler = SRS(np.arange(0, 10000))

    train_loader = data_utils.DataLoader(train_set, batch_size=batch_size, \
        sampler=train_sampler)
    val_loader = data_utils.DataLoader(val_set, batch_size=batch_size, \
        sampler=val_sampler)
    test_loader = data_utils.DataLoader(test_set, batch_size=batch_size, \
        sampler=test_sampler)
    
    return train_loader, val_loader, test_loader 

def get_cifar10_data_loaders(batch_size=64, n_train=40000, \
    n_val=10000, n_test=10000, train_transform=None, \
    val_transform=None, test_transform=None):

    assert n_train + n_val == 50000
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.,0.,0.), (1.,1.,1.))
        ])
    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.,0.,0.), (1.,1.,1.))
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.,0.,0.), (1.,1.,1.))
        ])
    
    train_set = datasets.CIFAR10(root=CIFAR10_DATA_ROOT, download=True, \
        train=True, transform=train_transform)
    val_set = datasets.CIFAR10(root=CIFAR10_DATA_ROOT, download=True, \
        train=True, transform=val_transform)
    test_set = datasets.CIFAR10(root=CIFAR10_DATA_ROOT, download=True, \
        train=False, transform=test_transform)

    indices = np.arange(0, 50000)
    np.random.seed(SEED)
    np.random.shuffle(indices)

    train_sampler = SRS(indices[:n_train])
    val_sampler = SRS(indices[n_train:])
    test_sampler = SRS(np.arange(0, 10000))

    train_loader = data_utils.DataLoader(train_set, batch_size=batch_size, \
        sampler=train_sampler)
    val_loader = data_utils.DataLoader(val_set, batch_size=batch_size, \
        sampler=val_sampler)
    test_loader = data_utils.DataLoader(test_set, batch_size=batch_size, \
        sampler=test_sampler)
    
    return train_loader, val_loader, test_loader 

def get_fmnist_data_loaders(batch_size=64, n_train=50000, \
    n_val=10000, n_test=10000, train_transform=None, \
    val_transform=None, test_transform=None):

    assert n_train + n_val == 60000
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    train_set = datasets.FashionMNIST(root=FMNIST_DATA_ROOT, download=True, \
        train=True, transform=train_transform)
    val_set = datasets.FashionMNIST(root=FMNIST_DATA_ROOT, download=True, \
        train=True, transform=val_transform)
    test_set = datasets.FashionMNIST(root=FMNIST_DATA_ROOT, download=True, \
        train=False, transform=test_transform)

    indices = np.arange(0, 60000)
    np.random.seed(SEED)
    np.random.shuffle(indices)

    train_sampler = SRS(indices[:n_train])
    val_sampler = SRS(indices[n_train:])
    test_sampler = SRS(np.arange(0, 10000))

    train_loader = data_utils.DataLoader(train_set, batch_size=batch_size, \
        sampler=train_sampler)
    val_loader = data_utils.DataLoader(val_set, batch_size=batch_size, \
        sampler=val_sampler)
    test_loader = data_utils.DataLoader(test_set, batch_size=batch_size, \
        sampler=test_sampler)
    
    return train_loader, val_loader, test_loader 

def progress(curr, total, suffix='', bar_len=48):
    filled = int(round(bar_len * curr / float(total))) if curr != 0 else 1
    bar = '=' * (filled - 1) + '>' + '-' * (bar_len - filled)
    sys.stdout.write('\r[%s](%d/%d) .. %s' % (bar, curr, total, suffix))
    sys.stdout.flush()
    if curr == total:
        bar = bar_len * '='
        sys.stdout.write('\r[%s](%d/%d) .. %s .. Completed\n' % (bar, curr, total, suffix))
    return 

def _preprocess_state_dict(state_dict):
    from collections import OrderedDict
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        if not k.endswith('num_batches_tracked'):
            if k.startswith('module.'):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
    return new_dict