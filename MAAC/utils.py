import torch
import numpy

def normal_init(m, mean, std):
    if isinstance(m, torch.nn.ConvTranspose2d) or isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def disable_grad(module):
    for param in module.parameters():
        param.requires_grad = False

def enable_grad(module):
    for param in module.parameters():
        param.requires_grad = True


