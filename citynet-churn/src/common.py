import pandas as pd
from sklearn.model_selection import train_test_split

NUM = ["monthly_bill", "tenure_months", "support_calls_90d", "outage_hours_90d", "age"]
CAT = ["region", "plan", "tv_bundle", "payment_method", "auto_pay", "paid_late_last_bill"]


def load_split():
    df = pd.read_csv("data/citynet_churn.csv")
    X = df[NUM + CAT]
    y = df["churned"]
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test