from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from utils import do_nothing, get_user_choice
import numpy as np
from preprocessing import normalize_minmax, normalize_zscore
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, ShuffleSplit
from dataclasses import dataclass

main_classifiers = {
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

def get_norm_input():
    norm_input_msg = f"Which normalization method? ({", ".join(normalization_methods)}, or all): "
    norm_input = (input(norm_input_msg)).lower()
    while norm_input not in normalization_methods and norm_input != "all":
        print("Invalid input, try again.")
        norm_input = (input(norm_input_msg)).lower()
    return norm_input

def get_best(score_dict):
    return max(score_dict, key=score_dict.get), max(score_dict.values())

def get_clf_input():
    clf_input_msg = f"Which classifier? ({", ".join(main_classifiers)}, or all): "
    clf_input = (input(clf_input_msg)).lower()
    while clf_input not in main_classifiers and clf_input != "all":
        print("Invalid input, try again.")
        clf_input = (input(clf_input_msg)).lower()
    return clf_input

def get_accuracy_msg(clf_name, clf_type, current_dataset, eval_method, normalization_method, score, knn_val=None, kfold_val=None):
    names = f"({current_dataset.capitalize()}, {eval_method.capitalize()}, {clf_name.capitalize()}, {normalization_method.capitalize()})"
    msgs = {
        "default": f"{names} Accuracy: {score:.2%}",
        "knn inner": f"{names} Accuracy using knn={knn_val}: {score:.2%}",
        "knn outer": f"{names} Best knn-value: {knn_val}, accuracy: {score:.2%}",
        "default kfold inner": f"{names} Accuracy using k-fold val of {kfold_val}: {score:.2%}",
        "default kfold outer": f"{names} Best kfold-value: {kfold_val}, accuracy: {score:.2%}",
        "knn kfold inner": f"{names} Accuracy using KNN value={knn_val} and K-fold value={kfold_val}: {score:.2%}",
        "knn kfold outer": f"{names} Best k-nearest-neighbor and kfold combo: KNN value={knn_val} and K-fold value={kfold_val}, accuracy: {score:.2%}"
    }
    return msgs[clf_type]

def classify_holdout(clf, clf_name, X, y, current_dataset, normalization_method):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf.fit(X_train, y_train)
    holdout_score = clf.score(X_test, y_test)
    print(get_accuracy_msg(
        clf_name=clf_name,
        clf_type="default",
        current_dataset=current_dataset,
        eval_method="holdout",
        normalization_method=normalization_method,
        score=holdout_score
    ))
    return holdout_score

def classify_knn_holdout(clf_name, X, y, current_dataset, normalization_method):
    k_vals = [3, 5, 7]
    k_scores = {}
    for k in k_vals:
        clf = KNeighborsClassifier(k)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        k_scores[k] = score
        print(get_accuracy_msg(
            clf_name=clf_name,
            current_dataset=current_dataset,
            clf_type="knn inner",
            eval_method="holdout",
            normalization_method=normalization_method,
            score=score,
            knn_val=k
        ))
    best_kval, best_score = get_best(k_scores)
    print(get_accuracy_msg(
        clf_name=clf_name,
        clf_type="knn outer",
        current_dataset=current_dataset,
        eval_method="holdout",
        normalization_method=normalization_method,
        score=best_score,
        knn_val=best_kval
    ))
    return k_scores

def classify_random_subsampling(clf, clf_name, X, y, current_dataset, normalization_method):
    scores = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    print(get_accuracy_msg(
        clf_type="default",
        current_dataset=current_dataset,
        eval_method="random subsampling",
        clf_name=clf_name,
        normalization_method=normalization_method,
        score=float(np.mean(scores))
        ))
    return scores

def classify_knn_random_subsampling(clf_name, X, y, current_dataset, normalization_method):
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
            clf_name=clf_name,
            current_dataset=current_dataset,
            clf_type="knn inner",
            eval_method="random subsampling",
            normalization_method=normalization_method,
            score=avg_score,
            knn_val=k
        ))
    best_kval, best_score = get_best(k_scores)
    print(get_accuracy_msg(
        clf_name=clf_name,
        clf_type="knn outer",
        current_dataset=current_dataset,
        eval_method="random subsampling",
        normalization_method=normalization_method,
        score=best_score,
        knn_val=best_kval
    ))
    return k_scores

def classify_split(clf, splitter, X, y):
    scores = []
    for train_index, test_index in splitter.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    return scores

def classify_kfold(clf, clf_name, X, y, current_dataset, norm_input):
    k_folds = [3, 5, 10]
    k_accuracies = {}
    for k in k_folds:
        splitter = KFold(n_splits=k, shuffle=True)
        scores = classify_split(clf, splitter, X, y)
        print(get_accuracy_msg(
            clf_name=clf_name,
            current_dataset=current_dataset,
            clf_type="default kfold inner",
            eval_method="kfold",
            normalization_method=norm_input,
            score=float(np.mean(scores)),
            kfold_val=k
        ))
        k_accuracies[k] = float(np.mean(scores))
    best_kval, best_score = get_best(k_accuracies)
    print(get_accuracy_msg(
            clf_name=clf_name,
            current_dataset=current_dataset,
            clf_type="default kfold outer",
            eval_method="kfold",
            normalization_method=norm_input,
            score=best_score,
            kfold_val=best_kval
        ))
    return k_accuracies

def classify_knn_kfold(clf, X, y, current_dataset, norm_input):
    knn_vals = [3, 5, 7]
    k_fold_vals = [3, 5, 10]
    knn_kfold_combos = {}
    for knn_val in knn_vals:
        clf = KNeighborsClassifier(n_neighbors=knn_val)
        for kfold_val in k_fold_vals:
            splitter = KFold(n_splits=kfold_val, shuffle=True)
            scores  = []
            for train_index, test_index in splitter.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                scores.append(clf.score(X_test, y_test))
            print(get_accuracy_msg(
            clf_name="k-nearest-neighbor",
            current_dataset=current_dataset,
            clf_type="knn kfold inner",
            eval_method="kfold",
            normalization_method=norm_input,
            score=float(np.mean(scores)),
            knn_val=knn_val,
            kfold_val=kfold_val
        ))
            knn_kfold_combos[(knn_val, kfold_val)] = float(np.mean(scores))
    best_combo, best_score = get_best(knn_kfold_combos)
    print(get_accuracy_msg(
        clf_name="k-nearest-neighbor",
        current_dataset=current_dataset,
        clf_type="knn kfold outer",
        eval_method="kfold",
        normalization_method=norm_input,
        score=best_score,
        knn_val=best_combo[0],
        kfold_val=best_combo[1]
    ))
    return knn_kfold_combos

evaluation_methods = {
    "holdout": [classify_holdout, classify_knn_holdout],
    "random subsampling": [classify_random_subsampling, classify_knn_random_subsampling],
    "kfold": [classify_kfold, classify_knn_kfold]
}

def get_evaluation_method():
    input_msg = f"Which normalization method? ({", ".join(evaluation_methods)}): "
    eval_method_input = (input(input_msg)).lower()
    while eval_method_input not in evaluation_methods:
        print("Invalid input, try again.")
        eval_method_input = (input(input_msg)).lower()
    return eval_method_input

def run_classifier(clf, clf_name, X, y, current_dataset, norm_input, eval_method_input):
    classify_default = evaluation_methods[eval_method_input][0]
    classify_knn = evaluation_methods[eval_method_input][1]
    if clf_name == "k-nearest-neighbor":
        classify_knn(clf_name, X, y, current_dataset, norm_input)
    else:
        classify_default(clf, clf_name, X, y, current_dataset, norm_input)
    
def classify_input(original_X, y, current_dataset, clf_input, norm_input, eval_method_input):
    if clf_input == "all":
        for clf_name, clf in main_classifiers.items():
            if norm_input == "all":
                for method, func in normalization_methods.items():
                    X = func(original_X)
                    run_classifier(clf, clf_name, X, y, current_dataset, method, eval_method_input)
            else:
                X = normalization_methods[norm_input](original_X)
                run_classifier(clf, clf_name, X, y, current_dataset, norm_input, eval_method_input)
    else:
        clf = main_classifiers[clf_input]
        if norm_input == "all":
            for method, func in normalization_methods.items():
                X = func(original_X)
                run_classifier(clf, clf_input, X, y, current_dataset, method, eval_method_input)
        else:
            X = normalization_methods[norm_input](original_X)
            run_classifier(clf, clf_input, X, y, current_dataset, norm_input, eval_method_input)

if __name__ == "__main__":
    from data import datasets_dict

    current_dataset = "blobs"
    X, y = datasets_dict[current_dataset]
    clf_name = "k-nearest-neighbor"
    normalization_method = "unnormalized"
    classify_knn_random_subsampling(clf_name, X, y, current_dataset, normalization_method)

