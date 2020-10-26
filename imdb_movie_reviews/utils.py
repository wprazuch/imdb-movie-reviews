import tensorflow_datasets as tfds
import numpy as np
import pandas as pd


def unpack_dataset(dataset):
    x, y = [], []
    for data, label in dataset.as_numpy_iterator():
        x.append(data)
        y.append(label)
    return x, y
