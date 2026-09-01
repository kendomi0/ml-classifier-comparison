from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from utils import do_nothing
import numpy as np
from preprocessing import normalize_minmax, normalize_zscore
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from dataclasses import dataclass
from typing import Optional

classifier_map = {
    "naive bayes": GaussianNB(),
    "decision tree": DecisionTreeClassifier(criterion="gini"),
    "support vector machine": SVC(decision_function_shape='ovo'),
    "artificial neural networks": MLPClassifier(hidden_layer_sizes=(25, 15), activation='tanh', max_iter=2000, random_state=42),
    "k-nearest-neighbor": None,
}

classifier_costs = {
    "naive bayes": 1,
    "decision tree": 1,
    "support vector machine": 2,
    "k-nearest-neighbor": 2,
    "artificial neural networks": 3,
}

normalization_methods = {
    "unnormalized": do_nothing,
    "minmax": normalize_minmax,
    "zscore": normalize_zscore
}

normalization_methods_costs = {
    "unnormalized": 0,
    "minmax": 1,
    "zscore": 1
}

combination_headings = {
    "some": "Top {number_of_combos} out of {total_results} total combinations: ",
    "all": "All combinations: ",
    "best": "Best combination: ",
    "only": "Combination: "
}

@dataclass
class ClassificationResult:
    classifier_name: str
    evaluation_method: str
    normalization_method: str
    dataset: str
    score: str
    knn_value: Optional[int] = None
    kfold_value: Optional[int] = None
    cost: Optional[int] = None

def get_best(score_dict):
    return max(score_dict, key=score_dict.get), max(score_dict.values())

def classify_holdout(classifier, classifier_name, X, y, current_dataset, normalization_method):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    classifier.fit(X_train, y_train)
    holdout_score = classifier.score(X_test, y_test)
    return ClassificationResult(
        classifier_name=classifier_name,
        evaluation_method="holdout",
        normalization_method=normalization_method,
        dataset=current_dataset,
        score=f"{holdout_score:.2%}"
    )
def classify_holdout_knn(classifier_name, X, y, current_dataset, normalization_method):
    k_vals = [3, 5, 7]
    k_scores = {}
    for k in k_vals:
        classifier = KNeighborsClassifier(k)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        k_scores[k] = score
    best_kval, best_score = get_best(k_scores)
    return ClassificationResult(
        classifier_name=classifier_name,
        evaluation_method="holdout",
        normalization_method=normalization_method,
        dataset=current_dataset,
        knn_value=best_kval,
        score=f"{best_score:.2%}"
    )

def classify_random_subsampling(classifier, classifier_name, X, y, current_dataset, normalization_method):
    scores = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        classifier.fit(X_train, y_train)
        scores.append(classifier.score(X_test, y_test))
    average = float(np.mean(scores))
    return ClassificationResult(
        classifier_name=classifier_name,
        evaluation_method="random subsampling",
        normalization_method=normalization_method,
        dataset=current_dataset,
        score=f"{average:.2%}"
    )

def classify_random_subsampling_knn(classifier_name, X, y, current_dataset, normalization_method):
    k_vals = [3, 5, 7]
    k_scores = {}
    for k in k_vals:
        scores = []
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
            k_next = KNeighborsClassifier(n_neighbors=k)
            k_next.fit(X_train, y_train)
            scores.append(k_next.score(X_test, y_test))
        avg_score = float(np.mean(scores))
        k_scores[k] = avg_score
    best_kval, best_score = get_best(k_scores)
    return ClassificationResult(
        classifier_name=classifier_name,
        evaluation_method="random subsampling",
        normalization_method=normalization_method,
        dataset=current_dataset,
        knn_value=best_kval,
        score=f"{best_score:.2%}"
    )

def classify_split(classifier, splitter, X, y):
    scores = []
    for train_index, test_index in splitter.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        scores.append(classifier.score(X_test, y_test))
    return scores

def classify_kfold(classifier, classifier_name, X, y, current_dataset, normalization_method):
    evaluation_method = "kfold"
    k_folds = [3, 5, 10]
    k_accuracies = {}
    for k in k_folds:
        splitter = KFold(n_splits=k, shuffle=True)
        scores = classify_split(classifier, splitter, X, y)
        k_accuracies[k] = float(np.mean(scores))
    best_kval, best_score = get_best(k_accuracies)
    return ClassificationResult(
        classifier_name=classifier_name,
        evaluation_method=evaluation_method,
        normalization_method=normalization_method,
        dataset=current_dataset,
        kfold_value=best_kval,
        score=f"{best_score:.2%}"
    )

def classify_kfold_knn(classifier_name, X, y, current_dataset, normalization_method):
    evaluation_method="kfold"
    knn_vals = [3, 5, 7]
    k_fold_vals = [3, 5, 10]
    knn_kfold_combos = {}
    for knn_val in knn_vals:
        classifier = KNeighborsClassifier(n_neighbors=knn_val)
        for kfold_val in k_fold_vals:
            splitter = KFold(n_splits=kfold_val, shuffle=True)
            scores = classify_split(classifier, splitter, X, y)
            knn_kfold_combos[(knn_val, kfold_val)] = float(np.mean(scores))
    best_combo, best_score = get_best(knn_kfold_combos)
    return ClassificationResult(
        classifier_name=classifier_name,
        evaluation_method=evaluation_method,
        normalization_method=normalization_method,
        dataset=current_dataset,
        knn_value=best_combo[0],
        kfold_value=best_combo[1],
        score=f"{best_score:.2%}"
    )

def classify_loo(classifier, classifier_name, X, y, current_dataset, normalization_method):
    evaluation_method = "leave-one-out"
    loo = LeaveOneOut()
    scores = classify_split(classifier, loo, X, y)
    avg_score = float(np.mean(scores))
    return ClassificationResult(
        classifier_name=classifier_name,
        evaluation_method=evaluation_method,
        normalization_method=normalization_method,
        dataset=current_dataset,
        score=f"{avg_score:.2%}"
    )


def classify_loo_knn(classifier_name, X, y, current_dataset, normalization_method):
    evaluation_method="leave-one-out"
    loo = LeaveOneOut()
    knn_vals = [3, 5, 7]
    k_scores = {}
    for k in knn_vals:
        classifier = KNeighborsClassifier(n_neighbors=k)
        scores = classify_split(classifier, loo, X, y)
        k_scores[k] = float(np.mean(scores))
    best_kval, best_score = get_best(k_scores)
    return ClassificationResult(
        classifier_name=classifier_name,
        evaluation_method=evaluation_method,
        normalization_method=normalization_method,
        dataset=current_dataset,
        knn_value=best_kval,
        score=f"{best_score:.2%}"
    )

evaluation_methods = {
    "holdout": [classify_holdout, classify_holdout_knn],
    "random subsampling": [classify_random_subsampling, classify_random_subsampling_knn],
    "kfold": [classify_kfold, classify_kfold_knn],
    "leave-one-out": [classify_loo, classify_loo_knn]
}

evaluation_methods_costs = {
    "holdout": 1,
    "random subsampling": 2,
    "kfold": 3,
    "leave-one-out": 4
}

def classify(classifier, classifier_name, X, y, current_dataset, normalization_method, evaluation_method, lst):
    classify_default = evaluation_methods[evaluation_method][0]
    classify_knn = evaluation_methods[evaluation_method][1]
    if classifier_name == "k-nearest-neighbor":
        result = classify_knn(classifier_name, X, y, current_dataset, normalization_method)
    else:
        result = classify_default(classifier, classifier_name, X, y, current_dataset, normalization_method)
    lst.append(result)
    return result

def run_classifier(original_X, y, current_dataset, classifier_name, normalization_method, evaluation_method):
    results = []
    #evaluation_method, classifier_name = check_valid_combination(evaluation_method, classifier_name)
    if evaluation_method == "all":
        evaluations = evaluation_methods.items()
    else:
        evaluations = [(evaluation_method, evaluation_methods[evaluation_method])]
    if classifier_name == "all":
        classifiers = classifier_map.items()
    else:
        classifiers = [(classifier_name, classifier_map[classifier_name])]
    if normalization_method == "all":
        normalizations = normalization_methods.items()
    else:
        normalizations = [(normalization_method, normalization_methods[normalization_method])]
    for evaluation in evaluations:
        for classifier in classifiers:
            for normalization in normalizations:
                normalization_method_name, normalizer = normalization
                clf_name, clf_func = classifier
                evaluation_method_name, _ = evaluation
                if evaluation_method_name == "leave-one-out" and clf_name == "artificial neural networks":
                    continue
                X = normalizer(original_X)
                classify(clf_func, clf_name, X, y, current_dataset, normalization_method_name, evaluation_method_name, results)
    return results

def calculate_cost(result=ClassificationResult):
    result.cost = 0
    result.cost += classifier_costs[result.classifier_name] + normalization_methods_costs[result.normalization_method] + evaluation_methods_costs[result.evaluation_method]
    return result.cost

def sort_by_cost(results):
    ranked_results = sorted(results, key=lambda result: result.score, reverse=True)
    results_with_recurring_scores = {}
    results_with_unique_scores = ranked_results.copy()
    scores = [float(result.score.strip("%")) for result in ranked_results]

    for index, score in enumerate(scores):
        count = scores.count(score)
        if count > 1:
            if score in results_with_recurring_scores:
                results_with_recurring_scores[score].append(ranked_results[index])
            else:
                results_with_recurring_scores[score] = [ranked_results[index]]
            results_with_unique_scores.remove(ranked_results[index])

    for results in results_with_recurring_scores.values():
        for result in results:
            result.cost = calculate_cost(result)
        results = sorted(results, key=lambda result:result.cost)

    unpacked_results_with_recurring_scores = [unpacked_result for result_list in results_with_recurring_scores.values() for unpacked_result in result_list]

    if len(results_with_unique_scores) == 0:
        sorted_results = unpacked_results_with_recurring_scores.copy()
    else:
        sorted_results = results_with_unique_scores + unpacked_results_with_recurring_scores
        sorted_results = sorted(sorted_results, key=lambda result: (-float(result.score.strip("%")), result.cost))
    return sorted_results

def display_combinations_heading(results, number_of_combos):
    heading = ""
    total_results = len(results)
    if number_of_combos == 1:
        if total_results == 1:
            heading = combination_headings["only"]
        else:
            heading = combination_headings["best"]
    elif total_results > number_of_combos:
        heading = combination_headings["some"].format(number_of_combos=number_of_combos, total_results=total_results)
    else:
        heading = combination_headings["all"]
    return heading

def format_combination(results):
    combos = []
    for index, result in enumerate(results):
        print_kfold_value = ""
        if result.kfold_value is not None:
            print_kfold_value = f" (Kfold value: {result.kfold_value})"
        print_knn_value = ""
        if result.knn_value is not None:
            print_knn_value = f" (KNN value: {result.knn_value})"
        display = f"{result.dataset.capitalize()}, {result.classifier_name.capitalize()}{print_knn_value}, {result.evaluation_method.capitalize()}{print_kfold_value}, {result.normalization_method.capitalize()}."
        display += f" Accuracy: {result.score}"
        if result.cost is not None:
            display += f", Computational cost: {result.cost}"
        if len(results) > 1:
            display = f"#{index+1}: {display}"
        combos.append(display)
    return combos

def rank_results(results):
    scores = [result.score for result in results]
    scores_without_duplicates = set(scores)
    if len(scores) != len(scores_without_duplicates):
        ranked_results = sort_by_cost(results)
    else:
        ranked_results = sorted(results, key=lambda result: result.score, reverse=True)
    return ranked_results

def display_selected_combos(results, number_of_combos):
    heading = display_combinations_heading(results, number_of_combos)
    combinations = []
    if len(results) == 1:
        combinations = format_combination(results)
    else:
        ranked_results = rank_results(results)
        if number_of_combos == 1:
            combinations = format_combination(ranked_results[:1])
        else:
            combinations = format_combination(ranked_results[:number_of_combos])
    return heading, combinations

