from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from utils import do_nothing, get_user_choice
import numpy as np
from preprocessing import normalize_minmax, normalize_zscore
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, ShuffleSplit

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
    return max(score_dict, key=score_dict.get)

def get_clf_input():
    clf_input_msg = f"Which classifier? ({", ".join(main_classifiers)}, or all): "
    clf_input = (input(clf_input_msg)).lower()
    while clf_input not in main_classifiers and clf_input != "all":
        print("Invalid input, try again.")
        clf_input = (input(clf_input_msg)).lower()
    return clf_input

def classify_holdout(clf, clf_name, X, y, current_dataset, normalization_method):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf.fit(X_train, y_train)
    holdout_score = clf.score(X_test, y_test)
    print(f"({current_dataset}) Holdout accuracy with {clf_name} ({normalization_method}): {round(holdout_score, 2)}")

def classify_knn_holdout(clf_name, X, y, current_dataset, normalization_method):
    k_vals = [3, 5, 7]
    k_scores = {}
    for k in k_vals:
        clf = KNeighborsClassifier(k)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        k_scores[k] = score
        print(f"({current_dataset}) Holdout accuracy with {clf_name} using k={k} ({normalization_method}): {round(score, 2)}")
    print(f"Best k-value for {current_dataset} holdout {normalization_method}: {get_best(k_scores)}")

def classify_random_subsampling(clf, clf_name, X, y, current_dataset, normalization_method):
    scores = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    print(f"({current_dataset}) Random subsampling average accuracy with {clf_name} ({normalization_method}): {round(np.mean(scores), 3)}")

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
        k_scores[k] = np.mean(scores)
        print(f"({current_dataset}) Random subsampling accuracy with {clf_name} using k={k} ({normalization_method}): {round(k_scores[k], 3)}")
    print(f"({current_dataset}) Best k-value for random subsampling {normalization_method}: {get_best(k_scores)}")

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
        print(f"({current_dataset})  Accuracy for k={k} {norm_input} NB: {round(float(np.mean(scores)), 3)}")
        k_accuracies[k] = float(np.mean(scores))
    best_k_val = max(k_accuracies, key=k_accuracies.get)
    print(f"({current_dataset}) {clf_name} Best value for k in k-fold cross-validation {norm_input}: {best_k_val}")
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
            print(f"({current_dataset}) Random subsampling accuracy with k-nearest-neighbor ({norm_input}) using KNN value={knn_val} and K-fold value {kfold_val}: {float(np.mean(scores))}")
            knn_kfold_combos[(knn_val, kfold_val)] = float(np.mean(scores))
    best_combo = max(knn_kfold_combos, key=knn_kfold_combos.get)
    print(f"({current_dataset}) Best combo with k-nearest-neighbor and kfold {norm_input} is: KNN value of {best_combo[0]} and K-fold value of {best_combo[1]}")
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
    import utils
    from data import datasets_dict

    """
    clf = main_classifiers["naive bayes"]
    current_dataset = "blobs"
    X, y = datasets_dict[current_dataset]
    classify_knn_kfold(current_dataset, X, y)
    current_dataset = utils.get_user_choice(datasets_dict)
    original_X, y = datasets_dict[current_dataset]
    norm_input = get_norm_input()
    clf_input = get_clf_input()
    eval_method_input = get_evaluation_method()
    classify_input(original_X, y, current_dataset, clf_input, norm_input, eval_method_input)
    """
