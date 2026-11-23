import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from common import load_split
from lab3_pipeline import make_preprocess
def train_model(X_train, y_train):
    pipe = Pipeline([("pre", make_preprocess()),
                     ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    return pipe.fit(X_train, y_train)
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_split()
    pipe = train_model(X_train, y_train)
    proba = pipe.predict_proba(X_val)[:, 1]
    print("PR-AUC:", round(average_precision_score(y_val, proba), 3))
    print(confusion_matrix(y_val, proba >= 0.5).tolist())
    print(classification_report(y_val, proba >= 0.5, digits=3))
    top = np.argsort(-proba)[:100]                # the call team's 100 calls
    hits = y_val.to_numpy()[top].sum()
    print("top-100 churners:", hits, "| random 100 would find about", round(100 * y_val.mean()))
