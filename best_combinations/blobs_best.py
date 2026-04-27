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

X, y = datasets.make_blobs(n_samples=1200, n_features=2, centers=3, cluster_std=1.2, center_box=(-10.0, 10.0), shuffle=True, random_state=42, return_centers=False)

minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X)

# KNN with holdout and minmax
k_vals = [3, 5, 7]
k_scores = {}
for k in k_vals:
    X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.3, random_state=42)
    k_next = KNeighborsClassifier(n_neighbors=k)
    k_next.fit(X_train, y_train)
    score = k_next.score(X_test, y_test)
    k_scores[k] = score
    print(f"Holdout accuracy for KNN minmax k={k}: {round(score, 3)}")

best_k = max(k_scores, key=k_scores.get)
print(f"Best k value for KNN with holdout minmax is {best_k} with accuracy {round(k_scores[best_k], 3)}")

# Scatter plot
create_scatter_plot(X, y, "blobs")