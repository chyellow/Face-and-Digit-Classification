import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

DIGIT_CONFIG = {
    "height": 28,
    "width": 28,
    "train_images": DATA_DIR / "digitdata" / "trainingimages",
    "train_labels": DATA_DIR / "digitdata" / "traininglabels",
    "val_images": DATA_DIR / "digitdata" / "validationimages",
    "val_labels": DATA_DIR / "digitdata" / "validationlabels",
    "test_images": DATA_DIR / "digitdata" / "testimages",
    "test_labels": DATA_DIR / "digitdata" / "testlabels",
}

FACE_CONFIG = {
    "height": 70,
    "width": 60,
    "train_images": DATA_DIR / "facedata" / "facedatatrain",
    "train_labels": DATA_DIR / "facedata" / "facedatatrainlabels",
    "val_images": DATA_DIR / "facedata" / "facedatavalidation",
    "val_labels": DATA_DIR / "facedata" / "facedatavalidationlabels",
    "test_images": DATA_DIR / "facedata" / "facedatatest",
    "test_labels": DATA_DIR / "facedata" / "facedatatestlabels",
}


def _parse_images(filepath, height, width):
    with open(filepath, "r") as f:
        raw = f.readlines()

    n_images = len(raw) // height
    images = np.zeros((n_images, height * width), dtype=np.float64)

    for i in range(n_images):
        block = raw[i * height : (i + 1) * height]
        for r, line in enumerate(block):
            for c in range(min(len(line.rstrip("\n")), width)):
                ch = line[c]
                if ch == "+":
                    images[i, r * width + c] = 1.0
                elif ch == "#":
                    images[i, r * width + c] = 2.0

    return images


def _parse_labels(filepath):
    with open(filepath, "r") as f:
        return np.array([int(line.strip()) for line in f if line.strip() != ""], dtype=np.int64)


def load_data(dataset):
    if dataset == "digits":
        cfg = DIGIT_CONFIG
    elif dataset == "faces":
        cfg = FACE_CONFIG
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use 'digits' or 'faces'.")

    h, w = cfg["height"], cfg["width"]

    train_images = _parse_images(cfg["train_images"], h, w)
    train_labels = _parse_labels(cfg["train_labels"])
    val_images = _parse_images(cfg["val_images"], h, w)
    val_labels = _parse_labels(cfg["val_labels"])
    test_images = _parse_images(cfg["test_images"], h, w)
    test_labels = _parse_labels(cfg["test_labels"])

    return {
        "train": (train_images, train_labels),
        "val": (val_images, val_labels),
        "test": (test_images, test_labels),
    }
