import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
import numpy as np

torch.manual_seed(420)
np.random.seed(420)

class fontGenerator(nn.Module):
    def __init__(self, input_dims):
        self.input_dims = input_dims

class fontDiscriminator(nn.Module):
    def __init__(self, input_dims):
        self.input_dims = input_dims
        