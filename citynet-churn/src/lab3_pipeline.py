from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from common import load_split, NUM, CAT
def make_preprocess():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUM), ("cat", categorical, CAT)])
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_split()
    pre = make_preprocess()
    Xt = pre.fit_transform(X_train)
    print(Xt.shape)
    print(list(pre.get_feature_names_out()[[0, 4, 5, 10, 13]]))
