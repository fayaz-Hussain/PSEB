"""
Project 1 - Little data, big model
Client: Courier company

The courier company wants a model that reads the handwritten digits that drivers
scribble on parcel labels. They only have a small number of labelled examples.

This script trains THREE models on the same small dataset so the learning curves
can be compared side by side:

    A. Underfit  - a model that is too small / too constrained to learn the task
    B. Overfit   - a big model trained for too long with no regularisation
    C. Fixed     - the same big model + Dropout + L2 + EarlyStopping
    D. Fixed + more data - model C trained on 600 images instead of 150, to show
                   that with little data the biggest lever is more labelled data

Outputs (written next to this file):
    curves.png    - loss and accuracy curves for all three models
    results.json  - final train / validation / test numbers for each model

Run:
    python train.py            # full run (~1 minute on a laptop CPU)
    python train.py --quick    # fewer epochs, for a smoke test
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow info spam

import matplotlib

matplotlib.use("Agg")  # no window needed, we only save a PNG
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow import keras

HERE = Path(__file__).resolve().parent
SEED = 42
N_TRAIN = 150  # "little data": 15 labelled examples per digit


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
def load_data(n_train: int = N_TRAIN, seed: int = SEED):
    """Return (x_train, y_train), (x_val, y_val), (x_test, y_test).

    Uses scikit-learn's built-in 8x8 handwritten digits (1,797 images) so the
    project runs offline. 30% is held out as a test set and ~30% as a
    validation set - these are IDENTICAL for every experiment. Training images
    are drawn from the remaining pool (~720 images); only `n_train` are used.
    """
    digits = load_digits()
    x = digits.images.astype("float32") / 16.0  # pixel values 0..16 -> 0..1
    y = digits.target.astype("int64")

    x_pool, x_test, y_pool, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=seed
    )
    x_pool, x_val, y_pool, y_val = train_test_split(
        x_pool, y_pool, test_size=0.43, stratify=y_pool, random_state=seed
    )
    if n_train < len(x_pool):
        x_pool, _, y_pool, _ = train_test_split(
            x_pool, y_pool, train_size=n_train, stratify=y_pool, random_state=seed
        )
    return (x_pool, y_pool), (x_val, y_val), (x_test, y_test)


# ----------------------------------------------------------------------------
# 2. Models
# ----------------------------------------------------------------------------
def build_underfit() -> keras.Model:
    """Far too small: a 2-neuron bottleneck cannot separate 10 digit classes."""
    model = keras.Sequential(
        [
            keras.Input(shape=(8, 8)),
            keras.layers.Flatten(),
            keras.layers.Dense(2, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ],
        name="A_underfit",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_big(regularised: bool) -> keras.Model:
    """A big network (~300k weights) for only 150 training images.

    regularised=False -> memorises the training set (overfitting)
    regularised=True  -> Dropout + L2 weight decay keep it honest
    """
    l2 = keras.regularizers.l2(1e-3) if regularised else None
    drop = 0.5 if regularised else 0.0

    layers: list[keras.layers.Layer] = [keras.Input(shape=(8, 8)), keras.layers.Flatten()]
    for units in (512, 512, 256):
        layers.append(keras.layers.Dense(units, activation="relu", kernel_regularizer=l2))
        if drop:
            layers.append(keras.layers.Dropout(drop))
    layers.append(keras.layers.Dense(10, activation="softmax"))

    model = keras.Sequential(layers, name="C_fixed" if regularised else "B_overfit")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ----------------------------------------------------------------------------
# 3. Training
# ----------------------------------------------------------------------------
def train_all(epochs: int):
    def early_stop():
        return keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=25, restore_best_weights=True
        )

    experiments = {
        "A. Underfit (tiny model)": dict(builder=build_underfit, callbacks=[], n=N_TRAIN),
        "B. Overfit (big model, no regularisation)": dict(
            builder=lambda: build_big(regularised=False), callbacks=[], n=N_TRAIN
        ),
        "C. Fixed (Dropout + L2 + EarlyStopping)": dict(
            builder=lambda: build_big(regularised=True), callbacks=[early_stop()], n=N_TRAIN
        ),
        "D. Fixed + more data (600 images)": dict(
            builder=lambda: build_big(regularised=True), callbacks=[early_stop()], n=600
        ),
    }

    histories, results = {}, {}
    for name, exp in experiments.items():
        (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = load_data(exp["n"])
        keras.utils.set_random_seed(SEED)  # same starting point for every model
        model = exp["builder"]()
        print(f"\n=== {name}: {model.count_params():,} parameters, "
              f"train={len(x_tr)} val={len(x_va)} test={len(x_te)} ===")
        hist = model.fit(
            x_tr,
            y_tr,
            validation_data=(x_va, y_va),
            epochs=epochs,
            batch_size=32,
            callbacks=exp["callbacks"],
            verbose=0,
        )
        h = hist.history
        _, train_acc = model.evaluate(x_tr, y_tr, verbose=0)
        _, test_acc = model.evaluate(x_te, y_te, verbose=0)
        best_epoch = int(np.argmin(h["val_loss"])) + 1

        histories[name] = h
        results[name] = {
            "parameters": int(model.count_params()),
            "training_images": int(len(x_tr)),
            "epochs_run": len(h["loss"]),
            "best_val_loss_epoch": best_epoch,
            "final_train_loss": round(float(h["loss"][-1]), 4),
            "final_val_loss": round(float(h["val_loss"][-1]), 4),
            "best_val_loss": round(float(min(h["val_loss"])), 4),
            "train_accuracy": round(float(train_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "diagnosis": diagnose(h),
        }
        r = results[name]
        print(
            f"epochs={r['epochs_run']}  train_acc={r['train_accuracy']:.3f}  "
            f"test_acc={r['test_accuracy']:.3f}  val_loss best={r['best_val_loss']:.3f} "
            f"(epoch {best_epoch}) final={r['final_val_loss']:.3f}\n-> {r['diagnosis']}"
        )
    return histories, results


def diagnose(h: dict) -> str:
    """Read the curves the way we did in class and say what is going on."""
    train_loss, val_loss = h["loss"][-1], h["val_loss"][-1]
    best_val = min(h["val_loss"])
    train_acc = h["accuracy"][-1]
    if train_acc < 0.80:
        return "UNDERFITTING: training accuracy is low - the model cannot even fit the data it sees."
    if val_loss > 1.5 * best_val and val_loss - train_loss > 0.2:
        return (
            "OVERFITTING: training loss keeps falling but validation loss climbed "
            f"{(val_loss / best_val - 1) * 100:.0f}% above its best - the model is memorising."
        )
    return "GOOD FIT: training and validation curves stay close together."


# ----------------------------------------------------------------------------
# 4. Plot
# ----------------------------------------------------------------------------
def plot_curves(histories: dict, results: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(21, 8.5), sharex="col")
    colors = {"train": "#2563eb", "val": "#ea580c"}

    for col, (name, h) in enumerate(histories.items()):
        epochs = np.arange(1, len(h["loss"]) + 1)
        r = results[name]

        ax = axes[0, col]
        ax.plot(epochs, h["loss"], color=colors["train"], label="train loss")
        ax.plot(epochs, h["val_loss"], color=colors["val"], label="validation loss")
        ax.axvline(r["best_val_loss_epoch"], color="grey", ls=":", lw=1, label="best val epoch")
        ax.set_title(f"{name}\n{r['parameters']:,} params, {r['training_images']} training images",
                     fontsize=11)
        ax.set_ylabel("loss")
        ax.set_ylim(0, 2.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[1, col]
        ax.plot(epochs, h["accuracy"], color=colors["train"], label="train acc")
        ax.plot(epochs, h["val_accuracy"], color=colors["val"], label="validation acc")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        ax.text(
            0.02,
            0.04,
            f"test acc = {r['test_accuracy']:.1%}",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="#999"),
        )

    fig.suptitle(
        f"Project 1 - Little data ({N_TRAIN} training images), big model: "
        "underfitting vs overfitting vs the fix (same validation & test sets)",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\nSaved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--quick", action="store_true", help="20 epochs smoke test")
    args = parser.parse_args()
    epochs = 20 if args.quick else args.epochs

    histories, results = train_all(epochs)
    plot_curves(histories, results, HERE / "curves.png")
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
