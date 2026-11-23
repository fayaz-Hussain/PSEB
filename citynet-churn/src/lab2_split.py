from common import load_split
X_train, X_val, X_test, y_train, y_val, y_test = load_split()
print(len(X_train), len(X_val), len(X_test))
print(round(y_train.mean(), 3), round(y_val.mean(), 3), round(y_test.mean(), 3))
