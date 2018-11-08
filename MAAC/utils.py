import torch
import numpy

def normal_init(m, mean, std):
    m.weight.data.normal_(mean, std)
    m.bias.data.zero_()

