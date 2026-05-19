import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import datasets_dict
import utils
from classifiers import get_classifier, get_normalization_method, run_classifier, get_evaluation_method, get_valid_number_of_combos, display_selected_combos
from plotting import create_scatter_plot

current_dataset = utils.get_user_choice(datasets_dict)

X, y = datasets_dict[current_dataset]

evaluation_method = get_evaluation_method()

classifier = get_classifier(evaluation_method)

normalization_method = get_normalization_method()

results = run_classifier(X, y, current_dataset, classifier, normalization_method, evaluation_method)

number_of_combos = get_valid_number_of_combos(results)
    
display_selected_combos(results, number_of_combos)

create_scatter_plot(X, y, current_dataset)
