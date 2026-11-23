import joblib
from common import load_split
from lab5_evaluate import train_model
X_train, X_val, X_test, y_train, y_val, y_test = load_split()
pipe = train_model(X_train, y_train)
joblib.dump(pipe, "models/churn_pipeline.joblib")
print("saved models/churn_pipeline.joblib")
