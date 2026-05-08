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

X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8)

# SVM with random subsampling and unnormalized

scores= []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"(Noisy circles) Average accuracy for random subsampling unnormalized (SVM): {round(np.mean(scores), 3)}")

# Scatter plot
create_scatter_plot(X, y, "noisy circles")