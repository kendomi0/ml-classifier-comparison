import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import datasets_dict
import utils
from preprocessing import normalize
from classifiers import get_classifier, get_normalization_method, run_classifier
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

classifier = get_classifier("kfold")

normalization_method = get_normalization_method()

run_classifier(X, y, current_dataset, classifier, normalization_method, "kfold")