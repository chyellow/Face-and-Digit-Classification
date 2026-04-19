import numpy as np


def sample_training_data(images, labels, fraction, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    n = len(labels)
    k = max(1, int(n * fraction))
    indices = rng.choice(n, size=k, replace=False)
    return images[indices], labels[indices]
