import torch
import numpy

def normal_init(m):
    if isinstance(m, torch.nn.ConvTranspose2d) or isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.normal(m.weight, mean=0., std=0.0001)
        m.bias.data.fill_(0.001)

def xavier_init(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

