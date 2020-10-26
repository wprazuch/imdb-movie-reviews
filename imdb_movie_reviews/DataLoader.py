import torch
from torchtext import data, datasets
import random

from sklearn.model_selection import KFold
import numpy as np


class DataLoader:

    def __init__(self, SEED=1):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

        self.TEXT = data.Field(tokenize='spacy')
        self.LABEL = data.LabelField(dtype=torch.float)

        self.train_data, self.test_data = datasets.IMDB.splits(self.TEXT, self.LABEL)
        self.SEED = SEED

    def get_data(self):
        return self.TEXT, self.LABEL, self.train_data, self.test_data
