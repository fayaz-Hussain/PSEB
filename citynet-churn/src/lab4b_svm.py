# Lab 4b: a linear SVM on CityNet.  Run: python src/lab4b_svm.py
import time
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from common import load_split, NUM, CAT
from lab3_pipeline import make_preprocess

X_train, X_val, X_test, y_train, y_val, y_test = load_split()

def no_scaling():
    # Lab 3 preprocessing with the StandardScaler removed
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT)])

def test(name, prep, model):
    pipe = Pipeline([("prep", prep), ("model", model)])
    start = time.time()
    cv = cross_validate(pipe, X_train, y_train, cv=5, scoring="average_precision")
    print(f"{name:24s} PR-AUC: {cv['test_score'].mean():.3f}   time: {time.time() - start:.0f}s")

# Part 1: compare
# TODO before running: guess which line takes the longest, and why
test("SVM linear", make_preprocess(), SVC(kernel="linear", class_weight="balanced"))
test("SVM linear, no scaling", no_scaling(), SVC(kernel="linear", class_weight="balanced"))

# Part 2: use it like Lab 5
pipe = Pipeline([("prep", make_preprocess()), ("model", SVC(kernel="linear", class_weight="balanced"))])
pipe.fit(X_train, y_train)
score = pipe.decision_function(X_val)
top = np.argsort(-score)[:100]
print("top-100 churners:", y_val.to_numpy()[top].sum(), "| random 100 would find about", round(100 * y_val.mean()))

# Part 3: which columns push a customer towards leaving?
w = pd.Series(pipe[-1].coef_[0], index=pipe[0].get_feature_names_out())
print("push to LEAVE:", w.nlargest(3).round(2).to_dict())
print("push to STAY: ", w.nsmallest(3).round(2).to_dict())
