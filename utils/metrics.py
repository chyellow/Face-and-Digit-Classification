import numpy as np


def accuracy(predictions, labels):
    return np.mean(predictions == labels)
