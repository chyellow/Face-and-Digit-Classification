from data_loader import load_data
from models.perceptron import Perceptron
from experiments import run_experiment


def main():
    print("=" * 60)
    print("DIGIT CLASSIFICATION - Perceptron")
    print("=" * 60)

    digit_data = load_data("digits")
    n_features = digit_data["train"][0].shape[1]

    run_experiment(
        model_cls=Perceptron,
        model_kwargs={"n_features": n_features, "n_classes": 10},
        data=digit_data,
    )

    print()
    print("=" * 60)
    print("FACE CLASSIFICATION - Perceptron")
    print("=" * 60)

    face_data = load_data("faces")
    n_features = face_data["train"][0].shape[1]

    run_experiment(
        model_cls=Perceptron,
        model_kwargs={"n_features": n_features, "n_classes": 2},
        data=face_data,
    )


if __name__ == "__main__":
    main()
