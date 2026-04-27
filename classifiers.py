from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from utils import do_nothing, get_user_choice
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

def get_x_input():
    X_input_msg = f"Which normalization method? ({", ".join(normalization_methods)}, or all): "
    X_input = (input(X_input_msg)).lower()
    while X_input not in normalization_methods and X_input != "all":
        print("Invalid input, try again.")
        X_input = (input(X_input_msg)).lower()
    return X_input

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
    
def classify_input(original_X, y, current_dataset, clf_input, X_input):
    if clf_input == "all":
        for clf_name, clf in main_classifiers.items():
            if X_input == "all":
                for method, func in normalization_methods.items():
                    X = func(original_X)
                    if clf_name == "k-nearest-neighbor":
                        classify_knn_holdout(clf_name, X, y, current_dataset, method)
                    else:
                        classify_holdout(clf, clf_name, X, y, current_dataset, method)
            else:
                X = normalization_methods[X_input](original_X)
                if clf_name == "k-nearest-neighbor":
                    classify_knn_holdout(clf_name, X, y, current_dataset, X_input)
                else:
                    classify_holdout(clf, clf_name, X, y, current_dataset, X_input)
    else:
        clf = main_classifiers[clf_input]
        if X_input == "all":
            for method, func in normalization_methods.items():
                X = func(original_X)
                if clf_input == "k-nearest-neighbor":
                    classify_knn_holdout(clf_input, X, y, current_dataset, method)
                else:
                    classify_holdout(clf, clf_input, X, y, current_dataset, method)
        else:
            X = normalization_methods[X_input](original_X)
            if clf_input == "k-nearest-neighbor":
                classify_knn_holdout(clf_input, X, y, current_dataset, X_input)
            else:
                classify_holdout(clf, clf_input, X, y, current_dataset, X_input)

