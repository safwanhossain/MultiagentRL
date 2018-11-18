import torch
import numpy

def normal_init(m, mean=0.0, std=0.02):
    if isinstance(m, torch.nn.ConvTranspose2d) or isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def xavier_init(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

