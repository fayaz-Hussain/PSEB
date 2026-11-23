# Deep Learning – Part 7 Practical Projects

Two completed real-world projects from **Part 7 · Projects**:

| # | Project | Client | What it shows | Folder |
|---|---------|--------|---------------|--------|
| 1 | **Little data, big model** | Courier company | Underfitting, overfitting and the fix, on real learning curves (`curves.png`) | [`project1_little_data_big_model/`](project1_little_data_big_model/) |
| 5 | **Live training dashboard** *(compulsory)* | City Hospital | Fix the curves live in Streamlit, then screen real (held-out) patients | [`project5_live_training_dashboard/`](project5_live_training_dashboard/) |

Each folder has its own README with the objectives, the workflow designed on paper first,
implementation details, how to run it, results with screenshots, and limitations.

## Repository layout

```
Assignment_3/
├── README.md                          ← you are here
├── requirements.txt                   ← one environment for both projects
├── project1_little_data_big_model/
│   ├── README.md
│   ├── train.py                       ← trains 3 models, saves curves.png + results.json
│   ├── curves.png
│   └── results.json
├── project5_live_training_dashboard/
│   ├── README.md
│   ├── app.py                         ← Streamlit dashboard
│   ├── trainer.py                     ← data, model, live diagnosis, screening metrics
│   └── screenshots/
└── scripts/
    └── capture_screenshots.py         ← drives the dashboard in a browser to make screenshots
```

## Quick start

Python 3.10 – 3.12.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r Assignment_3/requirements.txt

# Project 1
cd Assignment_3/project1_little_data_big_model
python train.py                # writes curves.png and results.json

# Project 5
cd ../project5_live_training_dashboard
streamlit run app.py           # opens http://localhost:8501
```

Both projects use datasets that ship with scikit-learn, so **no download or API key is needed**
and everything runs on a laptop CPU in about a minute.

## Tools

TensorFlow / Keras (models and callbacks) · scikit-learn (datasets, splits, scaling, metrics) ·
Streamlit (dashboard) · Matplotlib (static curves) · Playwright (automated screenshots).
