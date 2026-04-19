import time
import numpy as np
from utils.sampling import sample_training_data
from utils.metrics import accuracy


FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_TRIALS = 5


def run_experiment(model_cls, model_kwargs, data, fractions=None, n_trials=None):
    if fractions is None:
        fractions = FRACTIONS
    if n_trials is None:
        n_trials = N_TRIALS

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    results = {}

    for frac in fractions:
        trial_accs = []
        trial_errors = []
        trial_times = []

        for trial in range(n_trials):
            rng = np.random.default_rng(seed=trial)
            X_sub, y_sub = sample_training_data(X_train, y_train, frac, rng)

            model = model_cls(**model_kwargs)

            start = time.perf_counter()
            model.train(X_sub, y_sub, X_val, y_val)
            elapsed = time.perf_counter() - start

            preds = model.predict(X_test)
            acc = accuracy(preds, y_test)
            err = 1.0 - acc

            trial_accs.append(acc)
            trial_errors.append(err)
            trial_times.append(elapsed)

        mean_acc = np.mean(trial_accs)
        std_acc = np.std(trial_accs)
        mean_err = np.mean(trial_errors)
        std_err = np.std(trial_errors)
        mean_time = np.mean(trial_times)
        std_time = np.std(trial_times)

        results[frac] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "trial_accuracies": trial_accs,
            "mean_error": mean_err,
            "std_error": std_err,
            "trial_errors": trial_errors,
            "mean_time": mean_time,
            "std_time": std_time,
            "trial_times": trial_times,
        }

        print(
            f"  {frac * 100:5.1f}% data -> "
            f"accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})  "
            f"error: {mean_err:.4f} (+/- {std_err:.4f})  "
            f"train time: {mean_time:.4f}s (+/- {std_time:.4f}s)"
        )

    return results