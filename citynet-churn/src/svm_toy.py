# Lab A: your first SVM on 6 customers.  Run: python src/svm_toy.py
from sklearn.svm import SVC

X = [[60, 0], [48, 1], [40, 1],      # tenure, calls: stayed
     [5, 6],  [10, 4], [14, 5]]      # left
y = [0, 0, 0, 1, 1, 1]

model = SVC(kernel="linear").fit(X, y)

print("support vectors:", model.support_vectors_.tolist())
print("new customers:", model.predict([[50, 1], [8, 5], [30, 2]]).tolist())
print("scores:", model.decision_function([[50, 1], [8, 5], [30, 2]]).round(2).tolist())
