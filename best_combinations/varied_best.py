import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from plotting import create_scatter_plot

X, y = datasets.make_classification(n_samples=1500, n_features=2, n_redundant=0, n_informative=2, random_state=42, n_clusters_per_class=1)

zscale = StandardScaler()
X_zscale = zscale.fit_transform(X)

# ANN with kfold and zscore

k_folds = [3, 5, 10]
k_accuracies = {}
for k in k_folds:
    kf = KFold(n_splits=k, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X_zscale[train_index], X_zscale[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = MLPClassifier(hidden_layer_sizes=(25, 15), activation='tanh', max_iter=2000, random_state=42)
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    print(f"Accuracy for k={k} zscore ANN: {round(np.mean(scores), 3)}")
    k_accuracies[k] = np.mean(scores)
best_k_val = max(k_accuracies, key=k_accuracies.get)
print(f" Best value for k in k-fold cross-validation zscore (ANN): {best_k_val}")

# Scatter plot
create_scatter_plot(X, y, "varied")
