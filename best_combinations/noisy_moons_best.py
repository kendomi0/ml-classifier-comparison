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

X, y = datasets.make_moons(n_samples=1500, noise=0.05, shuffle=True, random_state=42)
zscore = StandardScaler()
X_zscore = zscore.fit_transform(X)

# KNN with holdout and zscore
# Holdout zscore (KNN)
k_vals = [3, 5, 7]
k_scores = {}
for k in k_vals:
    X_train, X_test, y_train, y_test = train_test_split(X_zscore, y, test_size=0.3, random_state=42)
    k_next = KNeighborsClassifier(n_neighbors=k)
    k_next.fit(X_train, y_train)
    score = k_next.score(X_test, y_test)
    k_scores[k] = score
    print(f"Holdout accuracy for KNN zscore k={k}: {round(score, 3)}")

best_k = max(k_scores, key=k_scores.get)
print(f"The best k value for KNN with holdout zscore is {best_k} with accuracy {round(k_scores[best_k], 3)}")

# Scatter plot
create_scatter_plot(X, y, "noisy moons")