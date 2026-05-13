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

normalization_methods = {
    "unnormalized": do_nothing,
    "minmax": normalize_minmax,
    "zscore": normalize_zscore
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

def get_best(score_dict):
    return max(score_dict, key=score_dict.get), max(score_dict.values())

def get_accuracy_msg(classifier_name, classifier_type, current_dataset, evaluation_method, normalization_method, score, knn_val=None, kfold_val=None):
    names = f"({current_dataset.capitalize()}, {evaluation_method.capitalize()}, {classifier_name.capitalize()}, {normalization_method.capitalize()})"
    msgs = {
        "default": f"{names} Accuracy: {score:.2%}",
        "knn inner": f"{names} Accuracy using knn={knn_val}: {score:.2%}",
        "knn outer": f"{names} Best knn-value: {knn_val}, accuracy: {score:.2%}",
        "default kfold inner": f"{names} Accuracy using k-fold val of {kfold_val}: {score:.2%}",
        "default kfold outer": f"{names} Best kfold-value: {kfold_val}, accuracy: {score:.2%}",
        "knn kfold inner": f"{names} Accuracy using KNN value={knn_val} and K-fold value={kfold_val}: {score:.2%}",
        "knn kfold outer": f"{names} Best k-nearest-neighbor and kfold combo: KNN value={knn_val} and K-fold value={kfold_val}, accuracy: {score:.2%}"
    }
    return msgs[classifier_type]

def classify_holdout(classifier, classifier_name, X, y, current_dataset, normalization_method):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    classifier.fit(X_train, y_train)
    holdout_score = classifier.score(X_test, y_test)
    print(get_accuracy_msg(
        classifier_name=classifier_name,
        classifier_type="default",
        current_dataset=current_dataset,
        evaluation_method="holdout",
        normalization_method=normalization_method,
        score=holdout_score
    ))
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
        print(get_accuracy_msg(
            classifier_name=classifier_name,
            current_dataset=current_dataset,
            classifier_type="knn inner",
            evaluation_method="holdout",
            normalization_method=normalization_method,
            score=score,
            knn_val=k
        ))
    best_kval, best_score = get_best(k_scores)
    print(get_accuracy_msg(
        classifier_name=classifier_name,
        classifier_type="knn outer",
        current_dataset=current_dataset,
        evaluation_method="holdout",
        normalization_method=normalization_method,
        score=best_score,
        knn_val=best_kval
    ))
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
    print(get_accuracy_msg(
        classifier_type="default",
        current_dataset=current_dataset,
        evaluation_method="random subsampling",
        classifier_name=classifier_name,
        normalization_method=normalization_method,
        score=average
        ))
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
        print(get_accuracy_msg(
            classifier_name=classifier_name,
            current_dataset=current_dataset,
            classifier_type="knn inner",
            evaluation_method="random subsampling",
            normalization_method=normalization_method,
            score=avg_score,
            knn_val=k
        ))
    best_kval, best_score = get_best(k_scores)
    print(get_accuracy_msg(
        classifier_name=classifier_name,
        classifier_type="knn outer",
        current_dataset=current_dataset,
        evaluation_method="random subsampling",
        normalization_method=normalization_method,
        score=best_score,
        knn_val=best_kval
    ))
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
        print(get_accuracy_msg(
            classifier_name=classifier_name,
            current_dataset=current_dataset,
            classifier_type="default kfold inner",
            evaluation_method=evaluation_method,
            normalization_method=normalization_method,
            score=float(np.mean(scores)),
            kfold_val=k
        ))
        k_accuracies[k] = float(np.mean(scores))
    best_kval, best_score = get_best(k_accuracies)
    print(get_accuracy_msg(
            classifier_name=classifier_name,
            current_dataset=current_dataset,
            classifier_type="default kfold outer",
            evaluation_method=evaluation_method,
            normalization_method=normalization_method,
            score=best_score,
            kfold_val=best_kval
        ))
    return ClassificationResult(
        classifier_name=classifier_name,
        evaluation_method=evaluation_method,
        normalization_method=normalization_method,
        dataset=current_dataset,
        kfold_value=best_kval,
        score=f"{best_score:.2%}"
    )

def classify_kfold_knn(X, y, current_dataset, normalization_method):
    classifier_name="k-nearest-neighbor"
    evaluation_method="kfold"
    knn_vals = [3, 5, 7]
    k_fold_vals = [3, 5, 10]
    knn_kfold_combos = {}
    for knn_val in knn_vals:
        classifier = KNeighborsClassifier(n_neighbors=knn_val)
        for kfold_val in k_fold_vals:
            splitter = KFold(n_splits=kfold_val, shuffle=True)
            scores = classify_split(classifier, splitter, X, y)
            print(get_accuracy_msg(
            classifier_name=classifier_name,
            current_dataset=current_dataset,
            classifier_type="knn kfold inner",
            evaluation_method=evaluation_method,
            normalization_method=normalization_method,
            score=float(np.mean(scores)),
            knn_val=knn_val,
            kfold_val=kfold_val
        ))
            knn_kfold_combos[(knn_val, kfold_val)] = float(np.mean(scores))
    best_combo, best_score = get_best(knn_kfold_combos)
    print(get_accuracy_msg(
        classifier_name=classifier_name,
        current_dataset=current_dataset,
        classifier_type="knn kfold outer",
        evaluation_method=evaluation_method,
        normalization_method=normalization_method,
        score=best_score,
        knn_val=best_combo[0],
        kfold_val=best_combo[1]
    ))
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
    print(get_accuracy_msg(
        classifier_name=classifier_name,
        classifier_type="default",
        current_dataset=current_dataset,
        evaluation_method=evaluation_method,
        normalization_method=normalization_method,
        score=avg_score
    ))
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
        print(get_accuracy_msg(
            classifier_name="k-nearest-neighbor",
            current_dataset=current_dataset,
            classifier_type="knn inner",
            evaluation_method="leave-one-out",
            normalization_method=normalization_method,
            score=float(np.mean(scores)),
            knn_val=k
        ))
    best_kval, best_score = get_best(k_scores)
    print(get_accuracy_msg(
        classifier_name="k-nearest-neighbor",
        current_dataset=current_dataset,
        classifier_type="knn outer",
        evaluation_method="leave-one-out",
        normalization_method=normalization_method,
        score=best_score,
        knn_val=best_kval
        )
    )
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

def get_classifier(evaluation_method):
    classifier_msg = f"Which classifier? ({", ".join(classifier_map)}, or all): "
    classifier = (input(classifier_msg)).lower()
    while (classifier not in classifier_map and classifier != "all") or (evaluation_method == "leave-one-out" and classifier == "artificial neural networks"):
        if evaluation_method == "leave-one-out" and classifier == "artificial neural networks":
            print("Artificial neural networks is not available to use with leave-one-out, due to the computational cost. Choose another classifier.")
        else:
            print("Invalid input, try again.")
        classifier = (input(classifier_msg)).lower()
    return classifier

def get_normalization_method():
    normalization_method_msg = f"Which normalization method? ({", ".join(normalization_methods)}, or all): "
    normalization_method = (input(normalization_method_msg)).lower()
    while normalization_method not in normalization_methods and normalization_method != "all":
        print("Invalid input, try again.")
        normalization_method = (input(normalization_method_msg)).lower()
    return normalization_method

def get_evaluation_method():
    input_msg = f"Which evaluation method? ({", ".join(evaluation_methods)}): "
    evaluation_method = (input(input_msg)).lower()
    while evaluation_method not in evaluation_methods:
        print("Invalid input, try again.")
        evaluation_method = (input(input_msg)).lower()
    return evaluation_method

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
    result_list = []
    if evaluation_method == "leave-one-out" and classifier_name == "artificial neural networks":
        raise ValueError("Leave-one-out and artificial neural networks is an invalid combination")
    elif classifier_name == "all":
        for clf_name, clf in classifier_map.items():
            if evaluation_method == "leave-one-out" and clf_name == "artificial neural networks":
                continue
            if normalization_method == "all":
                for method, func in normalization_methods.items():
                    X = func(original_X)
                    classify(clf, clf_name, X, y, current_dataset, method, evaluation_method, result_list)
            else:
                X = normalization_methods[normalization_method](original_X)
                classify(clf, clf_name, X, y, current_dataset, normalization_method, evaluation_method, result_list)
    else:
        classifier = classifier_map[classifier_name]
        if normalization_method == "all":
            for method, func in normalization_methods.items():
                X = func(original_X)
                classify(classifier, classifier_name, X, y, current_dataset, method, evaluation_method, result_list)
        else:
            X = normalization_methods[normalization_method](original_X)
            classify(classifier, classifier_name, X, y, current_dataset, normalization_method, evaluation_method, result_list)
    return result_list

if __name__ == "__main__":
    from data import datasets_dict
    from utils import get_user_choice

    current_dataset = get_user_choice(datasets_dict)
    original_X, y = datasets_dict[current_dataset]
    evaluation_method = get_evaluation_method()
    classifier_name = get_classifier(evaluation_method)
    normalization_method = get_normalization_method()
    lst = []

    print(run_classifier(
        original_X, y, current_dataset, classifier_name, normalization_method, evaluation_method
        )
        )