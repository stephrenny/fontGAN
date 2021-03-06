from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

"""
Dataset for loading in alphanumeric (abc..789) character images of given fonts
"""
class FontDataset(Dataset):
    """
    @ params
    fonts_directory (str): High level directory which contains subdirectories of fonts
        The subdirectories are named by their font name (e.g. Helvetica-Regular)
        Each directory contains images of lower and uppercase characters and numerals 0-9
        Images of 
    num_condition (int, tuple<int>): The number of characters to stack together as a condition
        If tuple, gives a low-high range for number of characters given
    """
    def __init__(self, fonts_directory, num_condition, regular_only=False):
        self.fonts_directory = fonts_directory
        self.num_condition = num_condition

        self.process_fonts(self.fonts_directory)

    def _process_fonts(self):
        self.fonts = []

    def _get_character_image(self, idx):
        pass

    def __len__(self):
        return self.length

    """
    Returns an font item of a certain font type (e.g. Helvetica-Reular)
    In the form of a condition, target tuple:
    condition: (n, w, h) tensor of n random character images
    target: (62, w, h) tensor of all the alphanumeric character images
    """
    def __getitem__(self, idx):
        pass

class CharacterDataset(Dataset):
    pass