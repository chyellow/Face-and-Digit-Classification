import numpy as np


class Perceptron:
    def __init__(self, n_features, n_classes, learning_rate=1.0, max_epochs=25):
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = np.zeros((n_classes, n_features))
        self.biases = np.zeros(n_classes)

    def predict_single(self, x):
        scores = self.weights @ x + self.biases
        return int(np.argmax(scores))

    def predict(self, X):
        scores = X @ self.weights.T + self.biases
        return np.argmax(scores, axis=1)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        n = len(y_train)
        best_weights = self.weights.copy()
        best_biases = self.biases.copy()
        best_val_acc = 0.0

        for epoch in range(self.max_epochs):
            order = np.random.permutation(n)

            for i in order:
                x, y_true = X_train[i], y_train[i]
                y_pred = self.predict_single(x)
                if y_pred != y_true:
                    self.weights[y_true] += self.learning_rate * x
                    self.biases[y_true] += self.learning_rate
                    self.weights[y_pred] -= self.learning_rate * x
                    self.biases[y_pred] -= self.learning_rate

            if X_val is not None:
                preds = self.predict(X_val)
                val_acc = np.mean(preds == y_val)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = self.weights.copy()
                    best_biases = self.biases.copy()

        if X_val is not None:
            self.weights = best_weights
            self.biases = best_biases
