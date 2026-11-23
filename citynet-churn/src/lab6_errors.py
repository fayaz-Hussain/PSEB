import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from common import load_split
from lab5_evaluate import train_model
X_train, X_val, X_test, y_train, y_val, y_test = load_split()
pipe = train_model(X_train, y_train)
imp = permutation_importance(pipe, X_val, y_val, scoring="average_precision",
                             n_repeats=10, random_state=42)
print(pd.Series(imp.importances_mean, index=X_val.columns).nlargest(4).round(3).to_dict())
val = X_val.assign(churned=y_val.to_numpy(), caught=pipe.predict(X_val) == 1)
churners = val[val.churned == 1]
print(churners.groupby("caught")[["tenure_months", "outage_hours_90d"]].mean().round(1))
print(churners.groupby("region")["caught"].mean().round(2).to_dict())
print("TEST PR-AUC, run once at the end:",
      round(average_precision_score(y_test, pipe.predict_proba(X_test)[:, 1]), 3))
