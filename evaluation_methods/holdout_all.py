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
zscore = StandardScaler()
X_zscore = zscore.fit_transform(X)

# CLASSIFIERS

print("### Naive Bayes Classifier")

# ### Naive bayes classifier

# Holdout unnormalized (NB)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = GaussianNB()
clf.fit(X_train, y_train)
holdout_score = clf.score(X_test, y_test)
print(f"({current_dataset}) Holdout accuracy unnormalized (NB): {round(holdout_score, 2)}")

# Holdout minmax (NB)
X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.3, random_state=42)
clf = GaussianNB()
clf.fit(X_train, y_train)
holdout_score = clf.score(X_test, y_test)
print(f"({current_dataset}) Holdout accuracy minmax (NB): {round(holdout_score, 2)}")

# Holdout zscore (NB)
X_train, X_test, y_train, y_test = train_test_split(X_zscore, y, test_size=0.3, random_state=42)
clf = GaussianNB()
clf.fit(X_train, y_train)
holdout_score = clf.score(X_test, y_test)
print(f"({current_dataset}) Holdout accuracy zscore (NB): {round(holdout_score, 2)}")


# ### Decision tree
print("### Decision tree")
# Holdout unnormalized (DT)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(criterion="gini")
clf.fit(X_train, y_train)
holdout_score = clf.score(X_test, y_test)
print(f"({current_dataset}) Holdout accuracy unnormalized (DT): {round(holdout_score, 2)}")

# Holdout minmax (DT)
X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(criterion="gini")
clf.fit(X_train, y_train)
holdout_score = clf.score(X_test, y_test)
print(f"({current_dataset}) Holdout accuracy  minmax (DT): {round(holdout_score, 2)}")

# Holdout zscore (DT)
X_train, X_test, y_train, y_test = train_test_split(X_zscore, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(criterion="gini")
clf.fit(X_train, y_train)
holdout_score = clf.score(X_test, y_test)
print(f"({current_dataset}) Holdout accuracy zscore (DT): {round(holdout_score, 2)}")



# ### Support vector machines
print("### Support vector machines")
# Holdout unnormalized (SVM)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = SVC(decision_function_shape='ovo')
clf.fit(X_train, y_train)
holdout_score = clf.score(X_test, y_test)
print(f"({current_dataset}) Holdout accuracy unnormalized (SVM): {round(holdout_score, 2)}")

# Holdout minmax (SVM)
X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.3, random_state=42)
clf = SVC(decision_function_shape='ovo')
clf.fit(X_train, y_train)
holdout_score = clf.score(X_test, y_test)
print(f"({current_dataset}) Holdout accuracy minmax (SVM): {round(holdout_score, 2)}")

# Holdout zscore (SVM)
X_train, X_test, y_train, y_test = train_test_split(X_zscore, y, test_size=0.3, random_state=42)
clf = SVC(decision_function_shape='ovo')
clf.fit(X_train, y_train)
holdout_score = clf.score(X_test, y_test)
print(f"({current_dataset}) Holdout accuracy zscore (SVM): {round(holdout_score, 2)}")


# ### K-nearest-neighbor
print("### K-nearest-neighbor")
# Holdout unnormalized (KNN)
k_vals = [3, 5, 7]
k_scores = {}
for k in k_vals:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    k_next = KNeighborsClassifier(n_neighbors=k)
    k_next.fit(X_train, y_train)
    score = k_next.score(X_test, y_test)
    k_scores[k] = score
    print(f"({current_dataset}) Holdout accuracy for KNN unnormalized k={k}: {round(score, 3)}")

best_k = max(k_scores, key=k_scores.get)
print(f"The best k value for KNN with holdout unnormalized is {best_k} with accuracy {round(k_scores[best_k], 3)}")

# Holdout minmax (KNN)
k_vals = [3, 5, 7]
k_scores = {}
for k in k_vals:
    X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.3, random_state=42)
    k_next = KNeighborsClassifier(n_neighbors=k)
    k_next.fit(X_train, y_train)
    score = k_next.score(X_test, y_test)
    k_scores[k] = score
    print(f"({current_dataset}) Holdout accuracy for KNN minmax k={k}: {round(score, 3)}")

best_k = max(k_scores, key=k_scores.get)
print(f"The best k value for KNN with holdout minmax is {best_k} with accuracy {round(k_scores[best_k], 3)}")

# Holdout zscore (KNN)
k_vals = [3, 5, 7]
k_scores = {}
for k in k_vals:
    X_train, X_test, y_train, y_test = train_test_split(X_zscore, y, test_size=0.3, random_state=42)
    k_next = KNeighborsClassifier(n_neighbors=k)
    k_next.fit(X_train, y_train)
    score = k_next.score(X_test, y_test)
    k_scores[k] = score
    print(f"({current_dataset}) Holdout accuracy for KNN zscore k={k}: {round(score, 3)}")

best_k = max(k_scores, key=k_scores.get)
print(f"The best k value for KNN with holdout zscore is {best_k} with accuracy {round(k_scores[best_k], 3)}")


# ### Artificial neural networks
print("### Artificial neural networks")
# Holdout unnormalized (ANN)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = MLPClassifier(hidden_layer_sizes=(25, 15), activation='tanh', max_iter=2000, random_state=42)
clf.fit(X_train, y_train)
holdout_score = clf.score(X_test, y_test)
print(f"({current_dataset}) Holdout accuracy unnormalized (ANN): {round(holdout_score, 2)}")

# Holdout minmax (ANN)
X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.3, random_state=42)
clf = MLPClassifier(hidden_layer_sizes=(25, 15), activation='tanh', max_iter=2000, random_state=42)
clf.fit(X_train, y_train)
holdout_score = clf.score(X_test, y_test)
print(f"({current_dataset}) Holdout accuracy minmax (ANN): {round(holdout_score, 2)}")

# Holdout zscore (ANN)
X_train, X_test, y_train, y_test = train_test_split(X_zscore, y, test_size=0.3, random_state=42)
clf = MLPClassifier(hidden_layer_sizes=(25, 15), activation='tanh', max_iter=2000, random_state=42)
clf.fit(X_train, y_train)
holdout_score = clf.score(X_test, y_test)
print(f"({current_dataset}) Holdout accuracy zscore (ANN): {round(holdout_score, 2)}")