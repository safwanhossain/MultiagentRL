import torch
import numpy

def normal_init(m, mean, std):
    if isinstance(m, torch.nn.ConvTranspose2d) or isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

