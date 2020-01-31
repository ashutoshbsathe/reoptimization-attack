from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable 
from resnet_model import resnet
from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack
from utils import progress, get_cifar10_data_loaders, _preprocess_state_dict
import numpy as np

model = resnet(num_classes=10, depth=110).cuda()
model.load_state_dict(_preprocess_state_dict(torch.load('./models/CIFAR10_PCL.pth.tar')['state_dict']))
model.eval()

sub = resnet(num_classes=10, depth=110).cuda()
sub.load_state_dict(_preprocess_state_dict(torch.load('./substitute_models/cifar_pcl_defence.pt')))
sub.eval()

adversaries = [
    GradientSignAttack(model, nn.CrossEntropyLoss(size_average=False), eps=float(8.0/255)),
    GradientSignAttack(sub, nn.CrossEntropyLoss(size_average=False), eps=float(8.0/255)),
]
_, _, test_loader = get_cifar10_data_loaders()
for adversary in adversaries:
    correct_adv = 0
    for i, (x_batch, y_batch) in enumerate(test_loader):
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        adv_x_batch = adversary.perturb(x_batch, y_batch)
        logits = model(adv_x_batch)
        _, preds = torch.max(logits, 1)
        correct_adv += (preds == y_batch).sum().item()
        progress(i+1, len(test_loader), 'correct_adv = {}'.format(correct_adv))
