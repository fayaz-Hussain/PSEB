import sys, joblib, pandas as pd
pipe = joblib.load("models/churn_pipeline.joblib")
new = pd.read_csv(sys.argv[1])
new["churn_risk"] = pipe.predict_proba(new)[:, 1].round(3)
out = new.sort_values("churn_risk", ascending=False)[["customer_id", "plan", "tenure_months", "support_calls_90d", "churn_risk"]]
out.to_csv("reports/predictions.csv", index=False)
print(out.to_string(index=False))
