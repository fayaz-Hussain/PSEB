import pandas as pd
df = pd.read_csv("data/citynet_churn.csv")
print(df.shape)
print(df.dtypes.value_counts().to_dict())
missing = df.isna().sum()
print(missing[missing > 0].to_dict())
print(df["churned"].value_counts(normalize=True).round(3).to_dict())
print(df[["tenure_months", "support_calls_90d", "outage_hours_90d"]].describe().loc[["mean", "min", "max"]].round(1))
print(df.groupby("support_calls_90d")["churned"].mean().round(2).head(6).to_dict())
print(df.groupby("cancel_request_logged")["churned"].mean().round(3).to_dict())

# TODO 1: Which column is missing values, and how many?
#payment_method has 149 missing values and age has 411
# TODO 2: What share of customers churned? Is the data balanced?
# No data is not balance the non churn has category 0.821 and churn has 0.179
# TODO 3: One column predicts churn almost perfectly. Which one, and why must we NOT use it?
#  cancel_request_logged perdicts the churn almost because it leakes the data 