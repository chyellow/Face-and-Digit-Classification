# Face and Digit Classification

Classifiers for handwritten digit recognition (0-9) and face detection (face vs. not face) using ASCII image datasets.

## Models

| Model | File | Status |
|-------|------|--------|
| Perceptron | `models/perceptron.py` | Done |
| Neural Network (from scratch) | `models/nn_scratch.py` | TODO |
| Neural Network (PyTorch) | `models/nn_pytorch.py` | TODO |

## Project Structure

```
data/
  digitdata/          # 28x28 ASCII digit images (0-9), 5000 train / 1000 val / 1000 test
  facedata/           # 70x60 ASCII face images (binary), 451 train / 301 val / 150 test
models/
  perceptron.py       # Multi-class perceptron with per-class weight vectors
  nn_scratch.py       # Three-layer neural network (numpy only)
  nn_pytorch.py       # Three-layer neural network (PyTorch)
utils/
  sampling.py         # Random subset sampling with seeded RNG
  metrics.py          # Accuracy computation
data_loader.py        # Parses ASCII images into numpy arrays, loads train/val/test splits
experiments.py        # Runs training across 10%-100% data fractions, 5 trials each
main.py               # Entry point - runs all experiments
```

## How to Run

```bash
python main.py
```

This trains each model on both datasets (digits and faces) at 10%, 20%, ..., 100% of the training data, running 5 trials per fraction. For each trial it reports:

- Mean accuracy and standard deviation
- Mean prediction error and standard deviation
- Mean training time and standard deviation

## How It Works

### Data Loading

`data_loader.py` reads the ASCII image files where each character maps to a feature value:
- Space = `0.0`
- `+` = `1.0`
- `#` = `2.0`

Each image is flattened into a 1D feature vector (784 features for digits, 4200 for faces).

```python
from data_loader import load_data

data = load_data("digits")  # or "faces"
X_train, y_train = data["train"]
X_val, y_val = data["val"]
X_test, y_test = data["test"]
```

### Experiment Runner

`experiments.py` handles the training loop across data fractions. Any model that implements `train(X, y, X_val, y_val)` and `predict(X)` plugs in directly:

```python
from experiments import run_experiment

results = run_experiment(
    model_cls=MyModel,
    model_kwargs={"n_features": 784, "n_classes": 10},
    data=data,
)
```

### Adding a New Model

1. Create a class with `train(X_train, y_train, X_val, y_val)` and `predict(X)` methods
2. Call `run_experiment()` with your class and kwargs
3. Results (accuracy, error, training time) are tracked automatically

## Requirements

- Python 3.10+
- NumPy
- PyTorch (for `nn_pytorch.py` only)
