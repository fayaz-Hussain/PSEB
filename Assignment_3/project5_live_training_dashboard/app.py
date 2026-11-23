"""
Project 5 - Live training dashboard
Client: City Hospital

Streamlit app with two jobs:
  1. TRAIN - tune the network in the sidebar, press Train, and watch the loss and
     accuracy curves draw live, epoch by epoch, with an automatic diagnosis
     (underfitting / overfitting / good fit). Fix the curves, then keep the model.
  2. SCREEN - use the trained model to screen held-out patients (or an uploaded
     CSV), choose the decision threshold, and see how many cancers are missed.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

import trainer
from trainer import PRESETS, TrainConfig

st.set_page_config(page_title="City Hospital - Live Training Dashboard", page_icon="🩺", layout="wide")

STATUS_STYLE = {
    "underfitting": ("🟡", st.warning),
    "overfitting": ("🔴", st.error),
    "good": ("🟢", st.success),
    "warming up": ("⏳", st.info),
}

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
DEFAULT_PRESET = "Overfit (big model, little data)"
WIDGET_KEYS = ["hidden_layers", "units", "dropout", "l2", "learning_rate", "epochs",
               "batch_size", "train_fraction", "early_stopping", "patience"]


def apply_preset() -> None:
    cfg = PRESETS[st.session_state.preset]
    for k in WIDGET_KEYS:
        st.session_state[k] = getattr(cfg, k)


if "preset" not in st.session_state:
    st.session_state.preset = DEFAULT_PRESET
    apply_preset()
st.session_state.setdefault("runs", [])        # history of every training run
st.session_state.setdefault("current", None)   # the model being used for screening


@st.cache_data(show_spinner=False)
def cached_data(train_fraction: float) -> trainer.Dataset:
    return trainer.load_data(train_fraction)


# ----------------------------------------------------------------------------
# Sidebar - the knobs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Training settings")
    st.selectbox("Start from a preset", list(PRESETS), key="preset", on_change=apply_preset,
                 help="Each preset reproduces one of the curve shapes from class.")
    st.caption("Change any knob below, then press **Train**.")

    st.subheader("Model size")
    st.slider("Hidden layers", 1, 4, key="hidden_layers")
    st.select_slider("Units per layer", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], key="units")

    st.subheader("Regularisation")
    st.slider("Dropout", 0.0, 0.6, step=0.05, key="dropout")
    st.select_slider("L2 weight decay", [0.0, 1e-4, 1e-3, 1e-2, 1e-1], key="l2",
                     format_func=lambda v: "off" if v == 0 else f"{v:g}")
    st.checkbox("Early stopping (restore best weights)", key="early_stopping")
    st.slider("Patience (epochs)", 5, 50, key="patience", disabled=not st.session_state.early_stopping)

    st.subheader("Optimisation & data")
    st.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2], key="learning_rate",
                     format_func=lambda v: f"{v:g}")
    st.slider("Max epochs", 10, 300, step=10, key="epochs")
    st.select_slider("Batch size", [4, 8, 16, 32, 64, 128], key="batch_size")
    st.slider("Fraction of training data used", 0.1, 1.0, step=0.05, key="train_fraction",
              help="Lower this to simulate 'little data' and provoke overfitting.")

    train_clicked = st.button("▶️ Train", type="primary", use_container_width=True)

cfg = TrainConfig(**{k: st.session_state[k] for k in WIDGET_KEYS})

# Allow screenshot automation: ?autotrain=1 trains once on first load.
if st.query_params.get("autotrain") == "1" and not st.session_state.get("_autotrained"):
    st.session_state._autotrained = True
    train_clicked = True

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🩺 City Hospital · Live Training Dashboard")
st.markdown(
    "Breast-cancer screening model trained on the **Wisconsin Diagnostic Breast Cancer** dataset "
    "(569 patients, 30 cell-nucleus measurements from a fine-needle biopsy image). "
    "**Positive class = malignant.** Fix the curves in **Train**, then use the model in **Screen patients**."
)

tab_train, tab_screen, tab_about = st.tabs(["📈 Train", "🔬 Screen patients", "ℹ️ About & limitations"])


# ----------------------------------------------------------------------------
# Live Keras callback -> Streamlit placeholders
# ----------------------------------------------------------------------------
class StreamlitLiveCurves(keras.callbacks.Callback):
    def __init__(self, max_epochs: int, ph_progress, ph_status, ph_loss, ph_acc):
        super().__init__()
        self.max_epochs = max_epochs
        self.ph_progress, self.ph_status, self.ph_loss, self.ph_acc = ph_progress, ph_status, ph_loss, ph_acc
        self.hist: dict[str, list[float]] = {}
        # Redraw at most ~60 times per run so the browser keeps up.
        self.every = max(1, max_epochs // 60)

    def on_epoch_end(self, epoch, logs=None):
        for k, v in (logs or {}).items():
            self.hist.setdefault(k, []).append(float(v))
        if (epoch + 1) % self.every == 0 or epoch + 1 == self.max_epochs:
            self.redraw(epoch + 1)

    def on_train_end(self, logs=None):
        self.redraw(len(self.hist.get("loss", [])), finished=True)

    def redraw(self, epoch: int, finished: bool = False):
        h = self.hist
        if not h:
            return
        idx = pd.RangeIndex(1, len(h["loss"]) + 1, name="epoch")
        self.ph_loss.line_chart(
            pd.DataFrame({"train loss": h["loss"], "validation loss": h["val_loss"]}, index=idx),
            color=["#2563eb", "#ea580c"], height=320,
        )
        self.ph_acc.line_chart(
            pd.DataFrame({"train accuracy": h["accuracy"], "validation accuracy": h["val_accuracy"]},
                         index=idx),
            color=["#2563eb", "#ea580c"], height=320,
        )
        label = "Finished" if finished else "Training"
        self.ph_progress.progress(min(epoch / self.max_epochs, 1.0),
                                  text=f"{label} · epoch {epoch}/{self.max_epochs}")
        render_status(self.ph_status, *trainer.diagnose(h))


def render_status(placeholder, status: str, why: str) -> None:
    icon, box = STATUS_STYLE[status]
    with placeholder.container():
        box(f"{icon} **{status.upper()}** — {why}")


def draw_static_curves(h: dict, col_loss, col_acc) -> None:
    idx = pd.RangeIndex(1, len(h["loss"]) + 1, name="epoch")
    col_loss.line_chart(pd.DataFrame({"train loss": h["loss"], "validation loss": h["val_loss"]}, index=idx),
                        color=["#2563eb", "#ea580c"], height=320)
    col_acc.line_chart(pd.DataFrame({"train accuracy": h["accuracy"],
                                     "validation accuracy": h["val_accuracy"]}, index=idx),
                       color=["#2563eb", "#ea580c"], height=320)


# ----------------------------------------------------------------------------
# TAB 1 - Train
# ----------------------------------------------------------------------------
with tab_train:
    data = cached_data(cfg.train_fraction)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training patients", len(data.y_train))
    c2.metric("Validation patients", len(data.y_val))
    c3.metric("Test patients (held out)", len(data.y_test))
    c4.metric("Model weights", f"{trainer.build_model(cfg, data.x_train.shape[1]).count_params():,}")

    ph_progress = st.empty()
    ph_status = st.empty()
    col_loss, col_acc = st.columns(2)
    col_loss.markdown("**Loss** (lower is better)")
    col_acc.markdown("**Accuracy**")
    ph_loss, ph_acc = col_loss.empty(), col_acc.empty()

    if train_clicked:
        live = StreamlitLiveCurves(cfg.epochs, ph_progress, ph_status, ph_loss, ph_acc)
        t0 = time.time()
        model, hist = trainer.train(cfg, data, extra_callbacks=[live])
        secs = time.time() - t0

        status, why = trainer.diagnose(hist)
        prob_val = model.predict(data.x_val, verbose=0).ravel()
        val_m = trainer.screening_metrics(data.y_val, prob_val, 0.5)
        run = {
            "run": len(st.session_state.runs) + 1,
            "preset": st.session_state.preset,
            **cfg.to_dict(),
            "epochs_run": len(hist["loss"]),
            "best_val_loss": round(min(hist["val_loss"]), 4),
            "final_val_loss": round(hist["val_loss"][-1], 4),
            "val_accuracy": round(val_m["accuracy"], 4),
            "val_sensitivity": round(val_m["sensitivity"], 4),
            "diagnosis": status,
            "seconds": round(secs, 1),
        }
        st.session_state.runs.append(run)
        st.session_state.current = {"model": model, "scaler": data.scaler, "history": hist,
                                    "run": run, "data": data, "status": (status, why)}
        st.toast(f"Run {run['run']} finished: {status}", icon=STATUS_STYLE[status][0])
    elif st.session_state.current:
        cur = st.session_state.current
        ph_progress.progress(1.0, text=f"Showing run {cur['run']['run']} "
                                       f"({cur['run']['epochs_run']} epochs) - change settings and press Train")
        render_status(ph_status, *cur["status"])
        draw_static_curves(cur["history"], ph_loss, ph_acc)
    else:
        ph_progress.info("👈 Pick a preset or set the knobs, then press **Train** to watch the curves draw live.")

    if st.session_state.runs:
        st.subheader("Run history")
        st.caption("Every run you train is logged here so you can compare what each change did.")
        runs_df = pd.DataFrame(st.session_state.runs).set_index("run")
        cols = ["preset", "hidden_layers", "units", "dropout", "l2", "learning_rate", "train_fraction",
                "early_stopping", "epochs_run", "best_val_loss", "final_val_loss", "val_accuracy",
                "val_sensitivity", "diagnosis"]
        st.dataframe(runs_df[cols], use_container_width=True)
        st.download_button("Download run history (CSV)", runs_df.to_csv().encode(), "run_history.csv",
                           "text/csv")

    with st.expander("How to read the curves"):
        st.markdown(
            "- **Underfitting** – both losses stay high and close together. The model is too weak. "
            "*Fix:* bigger model, higher learning rate, less regularisation, more epochs.\n"
            "- **Overfitting** – training loss keeps falling, validation loss turns and climbs. "
            "*Fix:* Dropout, L2, early stopping, smaller model, more data.\n"
            "- **Good fit** – both losses fall and level off together, with a small gap."
        )


# ----------------------------------------------------------------------------
# TAB 2 - Screen patients
# ----------------------------------------------------------------------------
with tab_screen:
    cur = st.session_state.current
    if cur is None:
        st.info("Train a model first (📈 Train tab). Use the **Fixed** preset for a model worth screening with.")
    else:
        model, scaler, d = cur["model"], cur["scaler"], cur["data"]
        status, _ = cur["status"]
        if status != "good":
            st.warning(f"The current model (run {cur['run']['run']}) was diagnosed as **{status}**. "
                       "Fix the curves before trusting it with patients.")
        else:
            st.success(f"Using model from run {cur['run']['run']} (diagnosis: good fit).")

        st.markdown("#### Decision threshold")
        threshold = st.slider(
            "Refer the patient for biopsy when P(malignant) ≥", 0.05, 0.95, 0.30, 0.05,
            help="In screening, a missed cancer (false negative) is far worse than an extra biopsy "
                 "(false positive), so the threshold is usually set below 0.5.",
        )

        prob_test = trainer.predict_patients(model, scaler, d.test_raw, d.feature_names)
        m = trainer.screening_metrics(d.y_test, prob_test, threshold)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Sensitivity (cancers caught)", f"{m['sensitivity']:.1%}")
        k2.metric("Specificity (healthy cleared)", f"{m['specificity']:.1%}")
        k3.metric("Accuracy", f"{m['accuracy']:.1%}")
        k4.metric("❌ Missed cancers (FN)", m["fn"])
        k5.metric("Extra biopsies (FP)", m["fp"])

        left, right = st.columns([1, 1.4])
        with left:
            st.markdown(f"**Confusion matrix** – {len(d.y_test)} held-out test patients")
            st.dataframe(pd.DataFrame(
                [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
                index=["Actually benign", "Actually malignant"],
                columns=["Predicted benign", "Predicted malignant"]), use_container_width=True)
        with right:
            st.markdown("**Threshold trade-off** (sensitivity vs specificity)")
            ts = np.round(np.arange(0.05, 0.96, 0.05), 2)
            sweep = pd.DataFrame([trainer.screening_metrics(d.y_test, prob_test, t) for t in ts], index=ts)
            sweep.index.name = "threshold"
            st.line_chart(sweep[["sensitivity", "specificity"]], color=["#dc2626", "#16a34a"], height=230)

        st.markdown("#### Screen one patient")
        pid = st.selectbox("Choose a held-out patient", d.test_raw["patient_id"])
        row = d.test_raw[d.test_raw["patient_id"] == pid]
        i = row.index[0]
        p = float(prob_test[i])
        truth = "Malignant" if d.y_test[i] == 1 else "Benign"
        refer = p >= threshold
        a, b, c = st.columns(3)
        a.metric("P(malignant)", f"{p:.1%}")
        b.metric("Recommendation", "Refer for biopsy" if refer else "Routine follow-up")
        c.metric("Actual diagnosis (for evaluation)", truth)
        if refer and truth == "Benign":
            st.caption("False positive: an unnecessary biopsy, but no cancer missed.")
        elif not refer and truth == "Malignant":
            st.error("False negative: this cancer would be MISSED at the current threshold.")
        key_feats = ["mean radius", "mean texture", "mean concave points", "worst radius",
                     "worst concave points", "worst area"]
        st.dataframe(row[["patient_id"] + key_feats].set_index("patient_id"), use_container_width=True)

        st.markdown("#### Screen a batch of patients")
        up = st.file_uploader("Upload a CSV with the 30 feature columns (one row per patient)", type="csv")
        if up is not None:
            batch = pd.read_csv(up)
            missing = [f for f in d.feature_names if f not in batch.columns]
            if missing:
                st.error(f"CSV is missing {len(missing)} required columns, e.g. {missing[:3]}")
                batch = None
        else:
            batch = d.test_raw.copy()
            st.caption("No file uploaded – showing all held-out test patients.")
        if batch is not None:
            bp = trainer.predict_patients(model, scaler, batch, d.feature_names)
            out = pd.DataFrame({
                "patient_id": batch["patient_id"] if "patient_id" in batch else np.arange(1, len(batch) + 1),
                "P(malignant)": np.round(bp, 3),
                "recommendation": np.where(bp >= threshold, "Refer for biopsy", "Routine follow-up"),
            })
            if up is None:
                out["actual"] = np.where(d.y_test == 1, "Malignant", "Benign")
                out["outcome"] = np.select(
                    [(bp >= threshold) & (d.y_test == 1), (bp < threshold) & (d.y_test == 1),
                     (bp >= threshold) & (d.y_test == 0)],
                    ["✅ caught", "❌ MISSED", "⚠️ extra biopsy"], "✅ cleared")
            st.dataframe(out.sort_values("P(malignant)", ascending=False), use_container_width=True,
                         hide_index=True, height=320)
            c_a, c_b = st.columns(2)
            c_a.download_button("Download screening results (CSV)", out.to_csv(index=False).encode(),
                                "screening_results.csv", "text/csv")
            c_b.download_button("Download a sample input CSV", d.test_raw.to_csv(index=False).encode(), "sample_patients.csv", "text/csv")


# ----------------------------------------------------------------------------
# TAB 3 - About
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
### Workflow (designed on paper first)
1. **Data** – 569 patients → stratified 60 / 20 / 20 split into train / validation / test.
   Scaler fitted on training rows only.
2. **Model** – small fully-connected network (Dense + ReLU → sigmoid), Adam, binary cross-entropy.
3. **Train live** – a Keras callback pushes each epoch's loss/accuracy into Streamlit charts.
4. **Diagnose** – rules read the curves: high train loss → underfitting; validation loss rising
   above its best while train loss falls → overfitting.
5. **Fix** – adjust size / Dropout / L2 / early stopping / learning rate / data and retrain.
6. **Screen** – apply the fixed model to *unseen* test patients, pick a threshold that favours
   sensitivity, report missed cancers.

### Limitations
- **Not a medical device.** For teaching only; it must not be used for real clinical decisions.
- Small, old, single-centre dataset (Wisconsin, 1990s); features are computed from biopsy images,
  so this does not replace a radiologist or pathologist.
- The test set has only ~114 patients, so each missed cancer moves sensitivity by ~2.4 points.
- No calibration, no external validation, no fairness / subgroup analysis.
- Models are kept in memory only; refreshing the page discards them.
"""
    )
