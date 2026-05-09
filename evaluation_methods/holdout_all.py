import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import datasets_dict
import utils
from classifiers import get_classifier, get_normalization_method, run_classifier

current_dataset = utils.get_user_choice(datasets_dict)

X, y = datasets_dict[current_dataset]

classifier = get_classifier("holdout")

normalization_method = get_normalization_method()

run_classifier(X, y, current_dataset, classifier, normalization_method, "holdout")
