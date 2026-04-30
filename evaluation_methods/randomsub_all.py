import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import datasets_dict
from utils import get_user_choice
from preprocessing import normalize
from classifiers import get_clf_input, get_x_input, get_evaluation_method, classify_input
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

clf_input = get_clf_input()
X_input = get_x_input()

classify_input(X, y, current_dataset, clf_input, X_input, "Random subsampling")