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

        for trial in range(n_trials):
            rng = np.random.default_rng(seed=trial)
            X_sub, y_sub = sample_training_data(X_train, y_train, frac, rng)

            model = model_cls(**model_kwargs)
            model.train(X_sub, y_sub, X_val, y_val)

            preds = model.predict(X_test)
            acc = accuracy(preds, y_test)
            trial_accs.append(acc)

        mean_acc = np.mean(trial_accs)
        std_acc = np.std(trial_accs)
        results[frac] = {"mean": mean_acc, "std": std_acc, "trials": trial_accs}

        print(f"  {frac*100:5.1f}% data -> accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")

    return results
