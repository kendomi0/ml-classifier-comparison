import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import datasets_dict
import utils
import preprocessing
from classifiers import get_clf_input, get_norm_input, classify_input, get_evaluation_method
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

current_dataset = utils.get_user_choice(datasets_dict)

X, y = datasets_dict[current_dataset]

clf_input = get_clf_input("holdout")

norm_input = get_norm_input()

classify_input(X, y, current_dataset, clf_input, norm_input, "holdout")