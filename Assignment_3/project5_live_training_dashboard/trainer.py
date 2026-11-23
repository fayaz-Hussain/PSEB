"""
Project 5 - Live training dashboard (core logic, no Streamlit code here).

Keeping the data, model and metric code in this file means it can be:
  * imported by the Streamlit app (app.py), and
  * run on its own from the command line as a smoke test:  python trainer.py
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

SEED = 42


# ----------------------------------------------------------------------------
# Config + presets
# ----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    hidden_layers: int = 2
    units: int = 32
    dropout: float = 0.2
    l2: float = 1e-3
    learning_rate: float = 1e-3
    epochs: int = 150
    batch_size: int = 16
    train_fraction: float = 1.0  # use less of the training set to simulate "little data"
    early_stopping: bool = True
    patience: int = 20

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, TrainConfig] = {
    # Too small and too slow: both curves stay high.
    "Underfit (too small, too slow)": TrainConfig(
        hidden_layers=1, units=1, dropout=0.0, l2=0.0, learning_rate=1e-4,
        epochs=60, batch_size=64, train_fraction=1.0, early_stopping=False,
    ),
    # Big model, little data, no brakes: train loss -> 0, validation loss climbs.
    "Overfit (big model, little data)": TrainConfig(
        hidden_layers=3, units=256, dropout=0.0, l2=0.0, learning_rate=3e-3,
        epochs=200, batch_size=8, train_fraction=0.25, early_stopping=False,
    ),
    # The fix: sensible size + Dropout + L2 + EarlyStopping, all the data.
    "Fixed (regularised + early stopping)": TrainConfig(
        hidden_layers=2, units=32, dropout=0.3, l2=1e-3, learning_rate=1e-3,
        epochs=200, batch_size=16, train_fraction=1.0, early_stopping=True, patience=20,
    ),
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@dataclass
class Dataset:
    feature_names: list[str]
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_raw: pd.DataFrame  # unscaled test patients, for the screening tab
    scaler: StandardScaler


def load_data(train_fraction: float = 1.0, seed: int = SEED) -> Dataset:
    """Wisconsin Diagnostic Breast Cancer data (569 patients, 30 features).

    Label is flipped so that 1 = MALIGNANT (the thing we must not miss) and
    0 = benign. Split 60/20/20 train/validation/test, stratified.
    The scaler is fitted on the training rows only (no leakage).
    """
    bunch = load_breast_cancer(as_frame=True)
    x = bunch.data
    y = (bunch.target == 0).astype("int64").to_numpy()  # 1 = malignant

    x_tr, x_rest, y_tr, y_rest = train_test_split(
        x, y, test_size=0.4, stratify=y, random_state=seed
    )
    x_va, x_te, y_va, y_te = train_test_split(
        x_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed
    )

    if train_fraction < 1.0:
        n = max(10, int(round(len(x_tr) * train_fraction)))
        x_tr, _, y_tr, _ = train_test_split(
            x_tr, y_tr, train_size=n, stratify=y_tr, random_state=seed
        )

    scaler = StandardScaler().fit(x_tr)
    test_raw = x_te.copy()
    test_raw.insert(0, "patient_id", [f"P{i:03d}" for i in range(1, len(test_raw) + 1)])
    test_raw = test_raw.reset_index(drop=True)

    return Dataset(
        feature_names=list(x.columns),
        x_train=scaler.transform(x_tr).astype("float32"),
        y_train=y_tr,
        x_val=scaler.transform(x_va).astype("float32"),
        y_val=y_va,
        x_test=scaler.transform(x_te).astype("float32"),
        y_test=y_te,
        test_raw=test_raw,
        scaler=scaler,
    )


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(cfg: TrainConfig, n_features: int) -> keras.Model:
    reg = keras.regularizers.l2(cfg.l2) if cfg.l2 > 0 else None
    model = keras.Sequential(name="screening_mlp")
    model.add(keras.Input(shape=(n_features,)))
    for _ in range(cfg.hidden_layers):
        model.add(keras.layers.Dense(cfg.units, activation="relu", kernel_regularizer=reg))
        if cfg.dropout > 0:
            model.add(keras.layers.Dropout(cfg.dropout))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Recall(name="recall")],
    )
    return model


def train(cfg: TrainConfig, data: Dataset, extra_callbacks=None, seed: int = SEED):
    """Train a model and return (model, history_dict)."""
    keras.utils.set_random_seed(seed)
    model = build_model(cfg, data.x_train.shape[1])
    callbacks = list(extra_callbacks or [])
    if cfg.early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=cfg.patience, restore_best_weights=True
            )
        )
    hist = model.fit(
        data.x_train,
        data.y_train,
        validation_data=(data.x_val, data.y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return model, hist.history


# ----------------------------------------------------------------------------
# Reading the curves
# ----------------------------------------------------------------------------
def diagnose(history: dict) -> tuple[str, str]:
    """Return (status, explanation). status is one of
    'underfitting', 'overfitting', 'good', 'warming up'."""
    loss = history.get("loss", [])
    val = history.get("val_loss", [])
    if len(loss) < 5:
        return "warming up", "Not enough epochs yet to judge the curves."

    acc = history["accuracy"][-1]
    best_val, best_ep = min(val), int(np.argmin(val)) + 1
    last_val, last_train = val[-1], loss[-1]
    gap = last_val - last_train

    if acc < 0.90 and last_train > 0.25:
        return (
            "underfitting",
            f"Training accuracy is only {acc:.0%} and training loss is still {last_train:.2f}. "
            "The model is too small, too constrained or learning too slowly. "
            "Try more units/layers, a higher learning rate, less L2/Dropout or more epochs.",
        )
    if last_val > 1.3 * best_val and gap > 0.1:
        return (
            "overfitting",
            f"Validation loss was best at epoch {best_ep} ({best_val:.3f}) and has since risen "
            f"to {last_val:.3f} while training loss fell to {last_train:.3f}. "
            "The model is memorising. Try Dropout, L2, early stopping, a smaller model or more data.",
        )
    return (
        "good",
        f"Train and validation loss track each other (gap {gap:+.3f}); best validation loss "
        f"{best_val:.3f} at epoch {best_ep}.",
    )


# ----------------------------------------------------------------------------
# Screening metrics
# ----------------------------------------------------------------------------
def screening_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": (tp + tn) / len(y_true),
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,  # recall on malignant
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def predict_patients(model: keras.Model, scaler: StandardScaler, raw: pd.DataFrame,
                     feature_names: list[str]) -> np.ndarray:
    """Probability of malignancy for raw (unscaled) patient rows."""
    x = scaler.transform(raw[feature_names]).astype("float32")
    return model.predict(x, verbose=0).ravel()


if __name__ == "__main__":
    # Command-line smoke test: train every preset and print the diagnosis.
    for name, cfg in PRESETS.items():
        d = load_data(cfg.train_fraction)
        model, h = train(cfg, d)
        status, why = diagnose(h)
        prob = model.predict(d.x_test, verbose=0).ravel()
        m = screening_metrics(d.y_test, prob, 0.5)
        print(f"\n{name}  ({len(h['loss'])} epochs, {len(d.y_train)} training patients)")
        print(f"  diagnosis: {status.upper()} - {why}")
        print(f"  test: accuracy={m['accuracy']:.3f} sensitivity={m['sensitivity']:.3f} "
              f"specificity={m['specificity']:.3f} missed cancers={m['fn']}")
