from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable 
from small_cnn import SmallCNN
from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack
from utils import progress, get_mnist_data_loaders
import numpy as np

model = SmallCNN().cuda()
model.load_state_dict(torch.load('./models/trades.pt'))
model.eval()

sub = SmallCNN().cuda()
sub.load_state_dict(torch.load('./substitute_models/mnist_trades.pt'))
sub.eval()

adversaries = [
    GradientSignAttack(model, nn.CrossEntropyLoss(size_average=False), eps=0.3),
    GradientSignAttack(sub, nn.CrossEntropyLoss(size_average=False), eps=0.3),
    LinfBasicIterativeAttack(model, nn.CrossEntropyLoss(size_average=False), eps=0.3, nb_iter=40, eps_iter=0.01),
    LinfBasicIterativeAttack(sub, nn.CrossEntropyLoss(size_average=False), eps=0.3, nb_iter=40, eps_iter=0.01)
]
_, _, test_loader = get_mnist_data_loaders()
for adversary in adversaries:
    correct_adv = 0
    for i, (x_batch, y_batch) in enumerate(test_loader):
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        adv_x_batch = adversary.perturb(x_batch, y_batch)
        logits = model(adv_x_batch)
        _, preds = torch.max(logits, 1)
        correct_adv += (preds == y_batch).sum().item()
        progress(i+1, len(test_loader), 'correct_adv = {}'.format(correct_adv))
