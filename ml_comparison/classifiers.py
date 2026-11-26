import time
import inspect
import warnings
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from ml_comparison.config import ML_COMPARISON_DEBUG
from ml_comparison.cost_calculations.cost_tiers import get_cost_tier

classifier_map = {
    "naive bayes": GaussianNB(),
    "decision tree": DecisionTreeClassifier(criterion="gini"),
    "support vector machine": SVC(decision_function_shape="ovo"),
    "artificial neural networks": MLPClassifier(
        hidden_layer_sizes=(25, 15),
        activation="tanh",
        max_iter=50,
        random_state=42,
        solver="lbfgs",
    ),
    "k-nearest-neighbor": None,
}

normalization_methods = {
    "unnormalized": FunctionTransformer(),
    "minmax": MinMaxScaler(),
    "zscore": StandardScaler(),
}

prohibited_combinations = [
    lambda evaluation_method, classifier, **_: evaluation_method == "leave-one-out" and classifier in ["artificial neural networks", "support vector machine"]
]

def is_prohibited(**kwargs: str):
    return any(combination(**kwargs) for combination in prohibited_combinations)

@dataclass
class ClassificationResult:
    classifier_name: str
    evaluation_method: str
    normalization_method: str
    dataset: str
    score: float
    knn_value: Optional[int] = None
    kfold_value: Optional[int] = None
    cost_tier: Optional[int] = None
    runtime: Optional[float] = None

def get_best(score_dict):
    return max(score_dict, key=score_dict.get), max(score_dict.values())


def accuracy_to_percent(score):
    new_score = score * (10**2)
    str_score = f"{new_score:.2f}"
    return float(str_score)


def classify_holdout(
    model, classifier_name, X, y, current_dataset, normalization_method
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model.fit(X_train, y_train)
    holdout_score = model.score(X_test, y_test)
    return ClassificationResult(
        classifier_name=classifier_name,
        evaluation_method="holdout",
        normalization_method=normalization_method,
        dataset=current_dataset,
        score=accuracy_to_percent(holdout_score),
    )


def classify_holdout_knn(classifier_name, X, y, current_dataset, normalization_method):
    k_vals = [3, 5, 7]
    k_scores = {}
    for k in k_vals:
        scaler = clone(normalization_methods[normalization_method])
        model = make_pipeline(scaler, KNeighborsClassifier(k))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        k_scores[k] = score
    best_kval, best_score = get_best(k_scores)
    return ClassificationResult(
        classifier_name=classifier_name,
        evaluation_method="holdout",
        normalization_method=normalization_method,
        dataset=current_dataset,
        knn_value=best_kval,
        score=accuracy_to_percent(best_score),
    )


def classify_random_subsampling(
    model, classifier_name, X, y, current_dataset, normalization_method
):
    scores = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    average = float(np.mean(scores))
    return ClassificationResult(
        classifier_name=classifier_name,
        evaluation_method="random subsampling",
        normalization_method=normalization_method,
        dataset=current_dataset,
        score=accuracy_to_percent(average),
    )


def classify_random_subsampling_knn(
    classifier_name, X, y, current_dataset, normalization_method
):
    k_vals = [3, 5, 7]
    k_scores = {}
    for k in k_vals:
        scores = []
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=i
            )
            scaler = clone(normalization_methods[normalization_method])
            model = make_pipeline(scaler, KNeighborsClassifier(k))
            model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))
        avg_score = float(np.mean(scores))
        k_scores[k] = avg_score
    best_kval, best_score = get_best(k_scores)
    return ClassificationResult(
        classifier_name=classifier_name,
        evaluation_method="random subsampling",
        normalization_method=normalization_method,
        dataset=current_dataset,
        knn_value=best_kval,
        score=accuracy_to_percent(best_score),
    )


def classify_split(classifier, splitter, X, y):
    scores = []
    for train_index, test_index in splitter.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        scores.append(classifier.score(X_test, y_test))
    return scores


def classify_kfold(
    model, classifier_name, X, y, current_dataset, normalization_method
):
    evaluation_method = "kfold"
    k_folds = [3, 5, 10]
    k_accuracies = {}
    for k in k_folds:
        splitter = KFold(n_splits=k, shuffle=True)
        scores = classify_split(model, splitter, X, y)
        k_accuracies[k] = float(np.mean(scores))
    best_kval, best_score = get_best(k_accuracies)
    return ClassificationResult(
        classifier_name=classifier_name,
        evaluation_method=evaluation_method,
        normalization_method=normalization_method,
        dataset=current_dataset,
        kfold_value=best_kval,
        score=accuracy_to_percent(best_score),
    )


def classify_kfold_knn(classifier_name, X, y, current_dataset, normalization_method):
    evaluation_method = "kfold"
    knn_vals = [3, 5, 7]
    k_fold_vals = [3, 5, 10]
    knn_kfold_combos = {}
    for knn_val in knn_vals:
        scaler = clone(normalization_methods[normalization_method])
        model = make_pipeline(scaler, KNeighborsClassifier(knn_val))
        for kfold_val in k_fold_vals:
            splitter = KFold(n_splits=kfold_val, shuffle=True)
            scores = classify_split(model, splitter, X, y)
            knn_kfold_combos[(knn_val, kfold_val)] = float(np.mean(scores))
    best_combo, best_score = get_best(knn_kfold_combos)
    return ClassificationResult(
        classifier_name=classifier_name,
        evaluation_method=evaluation_method,
        normalization_method=normalization_method,
        dataset=current_dataset,
        knn_value=best_combo[0],
        kfold_value=best_combo[1],
        score=accuracy_to_percent(best_score),
    )


def classify_loo(
    model, classifier_name, X, y, current_dataset, normalization_method
):
    evaluation_method = "leave-one-out"
    loo = LeaveOneOut()
    scores = classify_split(model, loo, X, y)
    avg_score = float(np.mean(scores))
    return ClassificationResult(
        classifier_name=classifier_name,
        evaluation_method=evaluation_method,
        normalization_method=normalization_method,
        dataset=current_dataset,
        score=accuracy_to_percent(avg_score),
    )


def classify_loo_knn(classifier_name, X, y, current_dataset, normalization_method):
    evaluation_method = "leave-one-out"
    knn_vals = [3, 5, 7]
    y = np.asarray(y)
    n_classes = y.max() + 1
    scaler = clone(normalization_methods[normalization_method])

    # Scale on all data at once due to the vectorization
    X = scaler.fit_transform(X)

    nn = NearestNeighbors(n_neighbors=max(knn_vals), n_jobs=-1).fit(X)
    _, idx = nn.kneighbors()
    neighbor_labels = y[idx]

    k_scores = {}
    for k in knn_vals:
        votes = (neighbor_labels[:, :k, None] == np.arange(n_classes)).sum(axis=1)
        preds = votes.argmax(axis=1)
        k_scores[k] = float((preds == y).mean())

    best_kval, best_score = get_best(k_scores)
    return ClassificationResult(
        classifier_name=classifier_name,
        evaluation_method=evaluation_method,
        normalization_method=normalization_method,
        dataset=current_dataset,
        knn_value=best_kval,
        score=accuracy_to_percent(best_score),
    )


evaluation_methods = {
    "holdout": [classify_holdout, classify_holdout_knn],
    "random subsampling": [
        classify_random_subsampling,
        classify_random_subsampling_knn,
    ],
    "kfold": [classify_kfold, classify_kfold_knn],
    "leave-one-out": [classify_loo, classify_loo_knn],
}


def classify(
    classifier_function,
    classifier_name,
    X,
    y,
    current_dataset,
    normalization_method,
    evaluation_method,
):
    classify_default = evaluation_methods[evaluation_method][0]
    classify_knn = evaluation_methods[evaluation_method][1]
    if classifier_name == "k-nearest-neighbor":
        result = classify_knn(
            classifier_name, X, y, current_dataset, normalization_method
        )
    else:
        scaler = clone(normalization_methods[normalization_method])
        model = make_pipeline(scaler, clone(classifier_function))
        result = classify_default(
            model, classifier_name, X, y, current_dataset, normalization_method
        )
    return result


def extract_selections(selected_options, all_options):
    if selected_options[0] == "all":
        return all_options.keys()
    return selected_options


def run_classifier(
    X,
    y,
    dataset: str,
    selected_classifiers: list,
    selected_normalization_methods: list,
    evaluation_method: str,
):
    results = []
    classifiers = extract_selections(selected_classifiers, classifier_map)
    normalizations = extract_selections(
        selected_normalization_methods, normalization_methods
    )
    is_one_combination = (
        len(classifiers) == 1 and len(normalizations) == 1
    )
    for classifier in classifiers:
        for normalization in normalizations:
            if is_prohibited(
                dataset=dataset,
                normalization_method=normalization,
                classifier=classifier,
                evaluation_method=evaluation_method
                ):
                if is_one_combination:
                    raise ValueError("The combination is prohibited")
                continue
            start_cpu_time, start_wall_clock_time = (
                time.process_time(),
                time.perf_counter(),
            )
            current_combination = f"{dataset}, {evaluation_method}, {classifier}, {normalization}"
            with warnings.catch_warnings(
                action="ignore", category=ConvergenceWarning
            ):
                result = classify(
                    classifier_function=classifier_map[classifier],
                    classifier_name=classifier,
                    X=X,
                    y=y,
                    current_dataset=dataset,
                    normalization_method=normalization,
                    evaluation_method=evaluation_method,
                )
            end_cpu_time = time.process_time() - start_cpu_time
            end_wall_clock_time = time.perf_counter() - start_wall_clock_time
            result.runtime = end_cpu_time
            result.cost_tier = get_cost_tier(result)
            results.append(result)
            if ML_COMPARISON_DEBUG:
                time_message = f"""
                    Combination: {current_combination}
                    CPU time (Runtime): {(1000*result.runtime):.3f} ms, {result.runtime:.3f} sec
                    Wall clock time: {1000*end_wall_clock_time:.2f} ms, {end_wall_clock_time:.3f} sec
                    ***
                """
                print(inspect.cleandoc(time_message))
    return results
