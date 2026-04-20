import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import datasets_dict
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

# Change this depending on which dataset you want to use
current_dataset = input("Type in which dataset you'd like to use: ")

X, y = datasets_dict[current_dataset]

# NORMALIZATION
# Min-max normalization
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X)
# Z-score normalization
zscale = StandardScaler()
X_zscale = zscale.fit_transform(X)

# CLASSIFIERS

# Naive Bayes classifier
print("### Naive bayes classifier")
# K fold unnormalized (NB)
k_folds = [3, 5, 10]
k_accuracies = {}
for k in k_folds:
    kf = KFold(n_splits=k, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    print(f"({current_dataset})  Accuracy for k={k} unnormalized NB: {round(np.mean(scores), 3)}")
    k_accuracies[k] = np.mean(scores)
best_k_val = max(k_accuracies, key=k_accuracies.get)
print(f"({current_dataset}) Best value for k in k-fold cross-validation unnormalized (NB): {best_k_val}")

# K fold min max (NB)
k_folds = [3, 5, 10]
k_accuracies = {}
for k in k_folds:
    kf = KFold(n_splits=k, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X_minmax[train_index], X_minmax[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    print(f"({current_dataset})  Accuracy for k={k} minmax NB: {round(np.mean(scores), 3)}")
    k_accuracies[k] = np.mean(scores)
best_k_val = max(k_accuracies, key=k_accuracies.get)
print(f"({current_dataset}) Best value for k in k-fold cross-validation minmax (NB): {best_k_val}")

# K fold z scale (NB)
k_folds = [3, 5, 10]
k_accuracies = {}
for k in k_folds:
    kf = KFold(n_splits=k, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X_zscale[train_index], X_zscale[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    print(f"({current_dataset})  Accuracy for k={k} zscore NB: {round(np.mean(scores), 3)}")
    k_accuracies[k] = np.mean(scores)
best_k_val = max(k_accuracies, key=k_accuracies.get)
print(f"({current_dataset})  Best value for k in k-fold cross-validation zscore (NB): {best_k_val}")

###


# Decision tree
print("### Decision tree")
clf = DecisionTreeClassifier(criterion="gini")
# K fold unnormalized (DT)
k_folds = [3, 5, 10]
k_accuracies = {}
for k in k_folds:
    kf = KFold(n_splits=k, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        scores.append(clf.score(X_test, y_test))
    print(f"({current_dataset})  Accuracy for k={k} unnormalized DT: {round(np.mean(scores), 3)}")
    k_accuracies[k] = np.mean(scores)
best_k_val = max(k_accuracies, key=k_accuracies.get)
print(f"({current_dataset}) Best value for k in k-fold cross-validation unnormalized (DT): {best_k_val}")

# K fold min max (DT)
k_folds = [3, 5, 10]
k_accuracies = {}
for k in k_folds:
    kf = KFold(n_splits=k, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X_minmax[train_index], X_minmax[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        scores.append(clf.score(X_test, y_test))
    print(f"({current_dataset})  Accuracy for k={k} minmax DT: {round(np.mean(scores), 3)}")
    k_accuracies[k] = np.mean(scores)
best_k_val = max(k_accuracies, key=k_accuracies.get)
print(f"({current_dataset}) Best value for k in k-fold cross-validation minmax (DT): {best_k_val}")

# K fold z scale (DT)
k_folds = [3, 5, 10]
k_accuracies = {}
for k in k_folds:
    kf = KFold(n_splits=k, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X_zscale[train_index], X_zscale[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        scores.append(clf.score(X_test, y_test))
    print(f"({current_dataset})  Accuracy for k={k} zscore DT: {round(np.mean(scores), 3)}")
    k_accuracies[k] = np.mean(scores)
best_k_val = max(k_accuracies, key=k_accuracies.get)
print(f"({current_dataset}) Best value for k in k-fold cross-validation zscore (DT): {best_k_val}")

# SVM
print("### Support vector machines")
# K fold unnormalized (SVM)
k_folds = [3, 5, 10]
k_accuracies = {}
for k in k_folds:
    kf = KFold(n_splits=k, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = SVC(decision_function_shape='ovo')
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    print(f"({current_dataset})  Accuracy for k={k} unnormalized SVM: {round(np.mean(scores), 3)}")
    k_accuracies[k] = np.mean(scores)
best_k_val = max(k_accuracies, key=k_accuracies.get)
print(f"({current_dataset}) Best value for k in k-fold cross-validation unnormalized (SVM): {best_k_val}")

# K fold min max (SVM)
k_folds = [3, 5, 10]
k_accuracies = {}
for k in k_folds:
    kf = KFold(n_splits=k, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X_minmax[train_index], X_minmax[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = SVC(decision_function_shape='ovo')
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    print(f"({current_dataset})  Accuracy for k={k} minmax SVM: {round(np.mean(scores), 3)}")
    k_accuracies[k] = np.mean(scores)
best_k_val = max(k_accuracies, key=k_accuracies.get)
print(f"({current_dataset}) Best value for k in k-fold cross-validation minmax (SVM): {best_k_val}")

# K fold z score (SVM)
k_folds = [3, 5, 10]
k_accuracies = {}
for k in k_folds:
    kf = KFold(n_splits=k, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X_zscale[train_index], X_zscale[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = SVC(decision_function_shape='ovo')
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        scores.append(clf.score(X_test, y_test))
    print(f"({current_dataset})  Accuracy for k={k} zscore SVM: {round(np.mean(scores), 3)}")
    k_accuracies[k] = np.mean(scores)
best_k_val = max(k_accuracies, key=k_accuracies.get)
print(f"({current_dataset}) Best value for k in k-fold cross-validation zscore (SVM): {best_k_val}")

# K-nearest-neighbor
print("### K-nearest-neighbor")
# K fold unnormalized (KNN)
k_folds = [3, 5, 10]
k_accuracies = {}
for k in k_folds:
    kf = KFold(n_splits=k, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        k_next = KNeighborsClassifier(n_neighbors=5)  # Fixed number of neighbors
        k_next.fit(X_train, y_train)
        scores.append(k_next.score(X_test, y_test))
    print(f"({current_dataset})  Accuracy for k={k} unnormalized KNN: {round(np.mean(scores), 3)}")
    k_accuracies[k] = np.mean(scores)
best_k_val = max(k_accuracies, key=k_accuracies.get)
print(f"({current_dataset}) Best value for k in k-fold cross-validation unnormalized (KNN): {best_k_val}")

# K fold min max (KNN)
k_folds = [3, 5, 10]
k_accuracies = {}
for k in k_folds:
    kf = KFold(n_splits=k, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X_minmax[train_index], X_minmax[test_index]
        y_train, y_test = y[train_index], y[test_index]
        k_next = KNeighborsClassifier(n_neighbors=5)  # Fixed number of neighbors
        k_next.fit(X_train, y_train)
        scores.append(k_next.score(X_test, y_test))
    print(f"({current_dataset})  Accuracy for k={k} minmax KNN: {round(np.mean(scores), 3)}")
    k_accuracies[k] = np.mean(scores)
best_k_val = max(k_accuracies, key=k_accuracies.get)
print(f"({current_dataset}) Best value for k in k-fold cross-validation minmax (KNN): {best_k_val}")

# K fold z scale (KNN)
k_folds = [3, 5, 10]
k_accuracies = {}
for k in k_folds:
    kf = KFold(n_splits=k, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X_zscale[train_index], X_zscale[test_index]
        y_train, y_test = y[train_index], y[test_index]
        k_next = KNeighborsClassifier(n_neighbors=5)  # Fixed number of neighbors
        k_next.fit(X_train, y_train)
        scores.append(k_next.score(X_test, y_test))
    print(f"({current_dataset})  Accuracy for k={k} zscore KNN: {round(np.mean(scores), 3)}")
    k_accuracies[k] = np.mean(scores)
best_k_val = max(k_accuracies, key=k_accuracies.get)
print(f"({current_dataset}) Best value for k in k-fold cross-validation zscore (KNN): {best_k_val}")

# Artificial neural networks
print("### Artificial neural networks")
# K fold unnormalized (ANN)
k_folds = [3, 5, 10]
k_accuracies = {}
for k in k_folds:
    kf = KFold(n_splits=k, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = MLPClassifier(hidden_layer_sizes=(25, 15), activation='tanh', max_iter=2000, random_state=42)
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    print(f"({current_dataset})  Accuracy for k={k} unnormalized ANN: {round(np.mean(scores), 3)}")
    k_accuracies[k] = np.mean(scores)
best_k_val = max(k_accuracies, key=k_accuracies.get)
print(f"({current_dataset}) Best value for k in k-fold cross-validation unnormalized (ANN): {best_k_val}")

# K fold min max (ANN)
k_folds = [3, 5, 10]
k_accuracies = {}
for k in k_folds:
    kf = KFold(n_splits=k, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X_minmax[train_index], X_minmax[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = MLPClassifier(hidden_layer_sizes=(25, 15), activation='tanh', max_iter=2000, random_state=42)
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    print(f"({current_dataset})  Accuracy for k={k} minmax ANN: {round(np.mean(scores), 3)}")
    k_accuracies[k] = np.mean(scores)
best_k_val = max(k_accuracies, key=k_accuracies.get)
print(f"({current_dataset}) Best value for k in k-fold cross-validation minmax(ANN): {best_k_val}")

# K fold z scale (ANN)
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
    print(f"({current_dataset})  Accuracy for k={k} zscore ANN: {round(np.mean(scores), 3)}")
    k_accuracies[k] = np.mean(scores)
best_k_val = max(k_accuracies, key=k_accuracies.get)
print(f"({current_dataset}) Best value for k in k-fold cross-validation zscale (ANN): {best_k_val}")