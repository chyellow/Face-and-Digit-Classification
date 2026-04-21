"""Template for a 3 layer feed forward neural network for digit
classification, implemented from scratch.

Architecture: input -> hidden1 -> hidden2 -> output.

Implement the forward pass, back propagation, and weight update
yourself. You may use numpy for linear algebra. You may not use torch,
tensorflow, sklearn, jax, or keras for training, gradients, or
prediction.

Required public API (fixed for auto grading):
  * class `ScratchNeuralNetworkDigits` with methods `forward`,
    `backward`, `update_weights`, `train`, `predict`, `evaluate`.
  * `main(training_percent: int, num_iterations: int = 5)`.

Usage:
    python3 q1b_neural_net_scratch_digits.py <training_percent>
"""

import sys
import time
import numpy as np

from util_digits import load_digits, flatten_images


class ScratchNeuralNetworkDigits:
    """3 layer fully connected network: 784 to h1 to h2 to 10.

    Use any reasonable hidden activation (ReLU, sigmoid, tanh). For the
    output layer, softmax paired with cross entropy loss is typical.
    Document your choices in the report.

    Implementation notes:
      * Store weights as numpy arrays: W1 (784, h1), W2 (h1, h2),
        W3 (h2, 10), plus biases b1, b2, b3.
      * Initialise with small random values (scaled Gaussian, He,
        Xavier) to break symmetry.
      * `forward` should cache intermediate activations so that
        `backward` can compute gradients without re running forward.
    """

    def __init__(
        self,
        input_size: int = 28 * 28,
        hidden1_size: int = 128,
        hidden2_size: int = 64,
        output_size: int = 10,
        learning_rate: float = 0.01,
        num_epochs: int = 20,
        batch_size: int = 32,
        seed: int | None = None,
    ):
        """Initialise network hyperparameters and weight matrices."""
        if seed is not None:
            np.random.seed(seed)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden1_size)
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0 / hidden1_size)
        self.b2 = np.zeros(hidden2_size)
        self.W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0 / hidden2_size)
        self.b3 = np.zeros(output_size)
        self.cache = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass.

        `X` has shape (N, 784). Return shape is (N, 10). You may return
        probabilities (after softmax) or raw logits; keep `predict` and
        `backward` consistent with your choice.
        """
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)
        z3 = a2 @ self.W3 + self.b3
        exp_z3 = np.exp(z3 - np.max(z3, axis=1, keepdims=True))
        y_hat = exp_z3 / np.sum(exp_z3, axis=1, keepdims=True)
        self.cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "y_hat": y_hat}
        return y_hat

    def backward(self, X: np.ndarray, y_onehot: np.ndarray) -> dict:
        """Back propagate loss gradients through the network.

        `X` has shape (N, 784); `y_onehot` has shape (N, 10). Return a
        dict like
        `{"dW1": ..., "db1": ..., "dW2": ..., "db2": ..., "dW3": ..., "db3": ...}`.
        """
        N = X.shape[0]
        y_hat = self.cache["y_hat"]
        a1 = self.cache["a1"]
        a2 = self.cache["a2"]
        z1 = self.cache["z1"]
        z2 = self.cache["z2"]

        dz3 = (y_hat - y_onehot) / N
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0)

        dz2 = (dz3 @ self.W3.T) * (z2 > 0).astype(float)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        dz1 = (dz2 @ self.W2.T) * (z1 > 0).astype(float)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}

    def update_weights(self, grads: dict) -> None:
        """Apply one gradient descent step using `grads` from `backward`."""
        self.W1 -= self.learning_rate * grads["dW1"]
        self.b1 -= self.learning_rate * grads["db1"]
        self.W2 -= self.learning_rate * grads["dW2"]
        self.b2 -= self.learning_rate * grads["db2"]
        self.W3 -= self.learning_rate * grads["dW3"]
        self.b3 -= self.learning_rate * grads["db3"]

    def train(self, training_images: np.ndarray, training_labels: np.ndarray) -> None:
        """Full training loop: epochs and mini batches.

        `training_images` has shape (N, 28, 28). `training_labels` has
        shape (N,) with values in {0..9}.
        """
        X = flatten_images(training_images)
        N = X.shape[0]
        y_onehot = np.zeros((N, 10))
        y_onehot[np.arange(N), training_labels] = 1.0

        for _ in range(self.num_epochs):
            perm = np.random.permutation(N)
            for start in range(0, N, self.batch_size):
                idx = perm[start:start + self.batch_size]
                X_batch = X[idx]
                y_batch = y_onehot[idx]
                self.forward(X_batch)
                grads = self.backward(X_batch, y_batch)
                self.update_weights(grads)

    def predict(self, image: np.ndarray) -> int:
        """Predict a single label in {0..9} for a 28x28 image."""
        X = image.flatten().reshape(1, -1)
        y_hat = self.forward(X)
        return int(np.argmax(y_hat, axis=1)[0])

    def evaluate(self, images: np.ndarray, labels: np.ndarray) -> float:
        """Return classification accuracy on a batch of images."""
        X = flatten_images(images)
        y_hat = self.forward(X)
        preds = np.argmax(y_hat, axis=1)
        return float(np.mean(preds == labels))


def main(training_percent: int, num_iterations: int = 5) -> dict:
    """Run the standard train/test pipeline for the scratch NN on digits."""
    training_images, training_labels = load_digits("training")
    test_images, test_labels = load_digits("test")

    num_total = len(training_images)
    sample_size = (num_total * training_percent) // 100

    train_times = np.zeros(num_iterations)
    accuracies = np.zeros(num_iterations)

    for i in range(num_iterations):
        idx = np.random.choice(num_total, size=sample_size, replace=False)
        x_sample = training_images[idx]
        y_sample = training_labels[idx]

        net = ScratchNeuralNetworkDigits()
        start = time.time()
        net.train(x_sample, y_sample)
        train_times[i] = time.time() - start

        accuracies[i] = net.evaluate(test_images, test_labels)

    errors = 1.0 - accuracies
    results = {
        "training_percent": training_percent,
        "mean_train_time": float(np.mean(train_times)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
    }

    print(f"\n=== Scratch NN | Digits | {training_percent}% of training data ===")
    print(f"Mean training time: {results['mean_train_time']:.3f} s")
    print(f"Mean accuracy:      {results['mean_accuracy']*100:.2f}%")
    print(f"Mean error:         {results['mean_error']*100:.2f}%")
    print(f"Std of error:       {results['std_error']*100:.2f}%")
    return results


if __name__ == "__main__":
    percent = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(percent)
