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

class FontDataset(Dataset):
    def __init__(self, fonts_directory, regular_only=False):
        self.fonts_directory = fonts_directory

        self.length = 0
        for font_name in os.listdir(fonts_directory):
            self.length += 1

    def _process_fonts(self):
        self.fonts = []

    def _get_character_image(self, idx):
        pass

    def __len__(self):
        return self.length

    # Shouldn't __getitem__ be giving back entire lists of characters at a time?
    def __getitem__(self, idx):
        pass