import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import datasets_dict
from utils import get_user_choice
from preprocessing import normalize
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

current_dataset = get_user_choice(datasets_dict)

X, y = datasets_dict[current_dataset]

X_minmax, X_zscore = normalize(X)

# CLASSIFIERS

# ### Naive bayes classifier
print("### Naive bayes classifier")
# Random subsampling unnormalized (NB)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling unnormalized (NB): {round(np.mean(scores), 3)}")
# Random subsampling minmax (NB)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.3)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling minmax (NB): {round(np.mean(scores), 3)}")
# Random subsampling zscore (NB)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_zscore, y, test_size=0.3)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling zscore (NB): {round(np.mean(scores), 3)}")


# ### Decision tree
print("### Decision tree")
# Random subsampling unnormalized (DT)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = DecisionTreeClassifier(criterion="gini")
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling unnormalized (DT): {round(np.mean(scores), 3)}")
# Random subsampling minmax (DT)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.3)
    clf = DecisionTreeClassifier(criterion="gini")
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling minmax (DT): {round(np.mean(scores), 3)}")
# Random subsampling zscore (DT)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_zscore, y, test_size=0.3)
    clf = DecisionTreeClassifier(criterion="gini")
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling zscore (DT): {round(np.mean(scores), 3)}")


# ### Support vector machines
print("### Support vector machines")
# Random subsampling unnormalized (SVM)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling unnormalized (SVM): {round(np.mean(scores), 3)}")
# Random subsampling minmax (SVM)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.3)
    clf = SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling minmax (SVM): {round(np.mean(scores), 3)}")
# Random subsampling zscore (SVM)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_zscore, y, test_size=0.3)
    clf = SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling zscore (SVM): {round(np.mean(scores), 3)}")


# ### K-nearest-neighbor
print("### K-nearest-neighbor")
# Random subsampling unnormalized (KNN)
k_vals = [3, 5, 7]
k_scores = {}
for k in k_vals:
    scores = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        k_next = KNeighborsClassifier(n_neighbors=k)
        k_next.fit(X_train, y_train)
        scores.append(k_next.score(X_test, y_test))
    k_scores[k] = np.mean(scores)
    print(f"({current_dataset}) Average accuracy for random subsampling unnormalized (KNN with k={k}): {round(k_scores[k], 3)}")

# Random subsampling minmax (KNN)
k_vals = [3, 5, 7]
k_scores = {}

for k in k_vals:
    scores = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.3, random_state=i)
        k_next = KNeighborsClassifier(n_neighbors=k)
        k_next.fit(X_train, y_train)
        scores.append(k_next.score(X_test, y_test))
    k_scores[k] = np.mean(scores)
    print(f"({current_dataset}) Average accuracy for random subsampling minmax (KNN with k={k}): {round(k_scores[k], 3)}")
# Random subsampling zscore (KNN)
k_vals = [3, 5, 7]
k_scores = {}
for k in k_vals:
    scores = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_zscore, y, test_size=0.3, random_state=i)
        k_next = KNeighborsClassifier(n_neighbors=k)
        k_next.fit(X_train, y_train)
        scores.append(k_next.score(X_test, y_test))
    k_scores[k] = np.mean(scores)
    print(f"({current_dataset}) Average accuracy for random subsampling zscore (KNN with k={k}): {round(k_scores[k], 3)}")


# ### Artificial neural networks
print("### Artificial neural networks")
# Random subsampling unnormalized (ANN)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = MLPClassifier(hidden_layer_sizes=(25, 15), activation='tanh', max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling unnormalized (ANN): {round(np.mean(scores), 3)}")
# Random subsampling minmax (ANN)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.3)
    clf = MLPClassifier(hidden_layer_sizes=(25, 15), activation='tanh', max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling minmax (ANN): {round(np.mean(scores), 3)}")
# Random subsampling zscore (ANN)
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_zscore, y, test_size=0.3)
    clf = MLPClassifier(hidden_layer_sizes=(25, 15), activation='tanh', max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
print(f"({current_dataset}) Average accuracy for random subsampling zscore (ANN): {round(np.mean(scores), 3)}")

