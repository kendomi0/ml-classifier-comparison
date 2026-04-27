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

base_x, base_y = datasets.make_blobs(n_samples=1500, centers=2, random_state=42,shuffle=True, center_box=(-10.0, 10.0))

# Adjusting anisotropic data to ensure the transformation is correctly applied
new_x = np.dot(base_x, [[0.6, -0.6], [-0.4, 0.8]])
X, y = new_x, base_y

minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X)

# Naive bayes with LOO and minmax
leave_one = LeaveOneOut()
scores = []
for train_index, test_index in leave_one.split(X_minmax):
    X_train, X_test = X_minmax[train_index], X_minmax[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"Leave-one-out accuracy minmax (NB): {round(np.mean(scores), 3)}")

# Scatter plot
create_scatter_plot(X, y, "anisotropic")
