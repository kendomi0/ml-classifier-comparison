from classifiers import get_norm_input, get_best, get_clf_input, classify_holdout, classify_knn_holdout, classify_input, main_classifiers, normalization_methods, classify_random_subsampling, main_classifiers, normalization_methods, classify_knn_random_subsampling, get_evaluation_method, evaluation_methods, run_classifier, classify_split, classify_kfold, classify_knn_kfold, get_accuracy_msg, classify_loo, classify_knn_loo
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from data import datasets_dict
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import pytest
from unittest.mock import MagicMock

@pytest.mark.parametrize("user_input", ["unnormalized", "minmax", "zscore"])
def test_get_norm_input_valid(monkeypatch, user_input):
    monkeypatch.setattr("builtins.input", lambda _: user_input)
    result = get_norm_input()
    assert result == user_input

def test_get_norm_input_invalid_then_valid(monkeypatch, capsys):
    inputs = iter(["invalid", "unnormalized"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    result = get_norm_input()
    captured = capsys.readouterr()
    assert "Invalid input" in captured.out
    assert result == "unnormalized"

def test_get_best():
    test_dict = {
        "Anna": 91,
        "Jenna": 76,
        "Lucy": 100,
        "Joe": 67
    }
    best_name, best_score = get_best(test_dict)
    assert best_name == "Lucy"
    assert best_score == 100

@pytest.mark.parametrize("user_input", ["naive bayes", "decision tree", "support vector machine", "artificial neural networks", "k-nearest-neighbor"])
def test_get_clf_input_valid(monkeypatch, user_input):
    monkeypatch.setattr("builtins.input", lambda _: user_input)
    eval_method_input="holdout"
    result = get_clf_input(eval_method_input)
    assert result == user_input

def test_get_clf_input_invalid_then_valid(monkeypatch, capsys):
    inputs = iter(["invalid", "decision tree"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    eval_method_input="holdout"
    result = get_clf_input(eval_method_input)
    captured = capsys.readouterr()
    assert "Invalid input" in captured.out
    assert result == "decision tree"

@pytest.mark.parametrize("user_input", ["nAIVE BAYES", "DECISion TReE", "SUppORt VecToR mAchInE", "ArtIfICial neURAL NETWORKS", "K-NEAREST-NEIGHBOR"])
def test_get_clf_input_valid_case_insensitive(monkeypatch, user_input):
    monkeypatch.setattr("builtins.input", lambda _: user_input)
    eval_method_input="holdout"
    result = get_clf_input(eval_method_input)
    assert result == user_input.lower()

def test_get_clf_input_loo_ann(monkeypatch):
    inputs = iter(["artificial neural networks", "naive bayes"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    eval_method_input = "leave-one-out"
    get_clf_input(eval_method_input)

@pytest.mark.parametrize("user_input",
    list(evaluation_methods.keys())
)
def test_get_evaluation_method_valid(monkeypatch, user_input):
    monkeypatch.setattr("builtins.input", lambda _: user_input)
    result = get_evaluation_method()
    assert result == user_input

@pytest.mark.parametrize("user_input",
    [m.upper() for m in evaluation_methods.keys()]
)
def test_get_evaluation_method_case_insensitive(monkeypatch, user_input):
    monkeypatch.setattr("builtins.input", lambda _: user_input)
    result = get_evaluation_method()
    assert result == user_input.lower()

def test_get_evaluation_method_invalid_valid(monkeypatch, capsys):
    inputs = iter(["invalid", list(evaluation_methods.keys())[0]])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    result = get_evaluation_method()
    captured = capsys.readouterr()
    assert "Invalid input" in captured.out
    assert result == list(evaluation_methods.keys())[0]

@pytest.mark.parametrize("clf_name, clf_type, eval_method, knn_val, kfold_val",
                       [
                           ("naive bayes", "default", "holdout", None, None),
                           ("k-nearest-neighbor", "knn inner", "random subsampling", 3, None),
                           ("k-nearest-neighbor", "knn outer", "holdout", 3, None),
                           ("naive bayes", "default kfold inner", "kfold", None, 3),
                           ("naive bayes", "default kfold outer", "kfold", None, 3),
                           ("k-nearest-neighbor", "knn kfold inner", "kfold", 3, 3),
                           ("k-nearest-neighbor", "knn kfold outer", "kfold", 3, 3)
                       ]
                       )
def test_get_accuracy_msg(clf_name, clf_type, eval_method, knn_val, kfold_val):
    current_dataset="blobs"
    normalization_method="unnormalized"
    score=0.98877
    result = get_accuracy_msg(clf_name, clf_type, current_dataset, eval_method, normalization_method, score, knn_val, kfold_val)
    print(result)
    terms = [clf_name, eval_method, current_dataset, normalization_method]
    new_terms = [term.capitalize() for term in terms]
    score_percent = "98.88%"
    new_terms.append(score_percent)
    for term in new_terms:
        assert term in result
    if knn_val is not None:
        assert str(knn_val) in result
    if kfold_val is not None:
        assert str(kfold_val) in result

def test_classify_holdout():
    X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8)
    clf = GaussianNB()
    clf_name = "naive bayes"
    current_dataset = "blobs"
    normalization_method = "unnormalized"
    score = classify_holdout(clf, clf_name, X, y, current_dataset, normalization_method)
    assert 0 < score <= 1.0

def test_classify_knn_holdout():
    X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8)
    clf_name = "k-nearest-neighbor"
    current_dataset = "blobs"
    normalization_method = "unnormalized"
    scores = classify_knn_holdout(clf_name, X, y, current_dataset, normalization_method)
    assert len(scores) == 3
    for score in scores.values():
        assert 0 < score <= 1.0

@pytest.mark.parametrize("current_dataset, normalization_method",
                       [
                           ("noisy circles", "unnormalized"),
                           ("noisy moons", "minmax"),
                           ("blobs", "zscore"),
                           ("varied", "unnormalized"),
                           ("varied", "minmax")
                       ]
                       )
def test_classify_random_subsampling(current_dataset, normalization_method):
    clf = MagicMock()
    clf_name = "naive bayes"
    clf.score.return_value = 0.9
    X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8)
    scores = classify_random_subsampling(clf, clf_name, X, y, current_dataset, normalization_method)
    assert len(scores) == 10

@pytest.mark.parametrize("current_dataset, normalization_method",
                        [
                            ("noisy circles", "unnormalized"),
                            ("noisy moons", "minmax"),
                            ("blobs", "zscore"),
                            ("anisotropic", "unnormalized"),
                            ("varied", "minmax")
                        ]
                        )
def test_classify_knn_random_subsampling(current_dataset, normalization_method):
    X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8)
    clf_name = "k-nearest-neighbor"
    scores = classify_knn_random_subsampling(clf_name, X, y, current_dataset, normalization_method)
    assert len(scores) == 3
    for score in scores.values():
        assert 0 < score <= 1.0

def test_classify_split():
    clf = main_classifiers["naive bayes"]
    current_dataset = "blobs"
    X, y = datasets_dict[current_dataset]
    splitter = KFold(n_splits=3, shuffle=True)
    result = classify_split(clf, splitter, X, y)
    assert isinstance(result, list)
    for score in result:
        assert 0 < score <= 1.0

def test_classify_kfold():
    clf_name = "naive bayes"
    clf = main_classifiers[clf_name]
    current_dataset = "blobs"
    norm_input = "unnormalized"
    X, y = datasets_dict[current_dataset]
    k_accuracies = classify_kfold(clf, clf_name, X, y, current_dataset, norm_input)
    assert len(k_accuracies) == 3
    for accuracy in k_accuracies.values():
        assert 0 < accuracy <= 1.0

def test_classify_knn_kfold():
    current_dataset = "blobs"
    norm_input = "minmax"
    X, y = datasets_dict[current_dataset]
    combo_scores = classify_knn_kfold(X, y, current_dataset, norm_input)
    assert len(combo_scores) == 9
    for accuracy in combo_scores.values():
        assert 0 < accuracy <= 1.0

def test_classify_loo():
    current_dataset = "blobs"
    X, y = datasets_dict[current_dataset]
    clf_name = "naive bayes"
    clf = main_classifiers[clf_name]
    norm_input = "unnormalized"
    score = classify_loo(clf, clf_name, X, y, current_dataset, norm_input)
    assert 0 < score <= 1

def test_classify_knn_loo(mocker):
    mock_split = mocker.patch("classifiers.classify_split")
    mock_split.return_value = [0.9, 0.85, 0.88]
    current_dataset = "blobs"
    X, y = datasets_dict[current_dataset]
    norm_input = "unnormalized"
    classify_knn_loo("k-nearest-neighbor", X, y, current_dataset, norm_input)
    assert any(
        isinstance(call.args[1], LeaveOneOut) for call in mock_split.call_args_list
        )
    
@pytest.mark.parametrize("clf_name, current_dataset, norm_input, eval_method_input",
                        [
                            ("naive bayes", "blobs", "unnormalized", "holdout"),
                            ("decision tree", "anisotropic", "minmax", "random subsampling"),
                            ("naive bayes", "blobs", "unnormalized", "kfold"),
                            ("decision tree", "blobs", "zscore", "leave-one-out"),
                            ("k-nearest-neighbor", "blobs", "unnormalized", "holdout"),
                            ("k-nearest-neighbor", "varied", "zscore", "random subsampling"),
                            ("k-nearest-neighbor", "blobs", "unnormalized", "kfold"),
                            ("k-nearest-neighbor", "blobs", "unnormalized", "leave-one-out")
                        ]
                        )
def test_run_classifer(clf_name, current_dataset, norm_input, eval_method_input, mocker):
    mock_dict = mocker.patch("classifiers.evaluation_methods")
    clf = main_classifiers[clf_name]
    X, y = datasets_dict[current_dataset]
    run_classifier(clf, clf_name, X, y, current_dataset, norm_input, eval_method_input)
    if clf_name != "k-nearest-neighbor":
        mock_dict[eval_method_input][0].assert_called_once()
    else:
        mock_dict[eval_method_input][1].assert_called_once()

@pytest.mark.parametrize("eval_method_input",
                         ["holdout", "random subsampling", "kfold", "leave-one-out"]
                         )
def test_classify_input_all_classifiers_all_normalization(eval_method_input, mocker):
    current_dataset = "blobs"
    X, y = datasets_dict[current_dataset]
    mock_run = mocker.patch("classifiers.run_classifier")
    classify_input(X, y, current_dataset, "all", "all", eval_method_input)

    clfs = list(main_classifiers)
    clfs_no_ann = [clf for clf in clfs if clf != "artificial neural networks"]
    norm_methods = list(normalization_methods)

    if eval_method_input == "leave-one-out":
        assert all(
            any(
                call.args[1] == clf_name and
                call.args[5] == norm_input
                for call in mock_run.call_args_list
            )
            for clf_name in clfs_no_ann
            for norm_input in norm_methods
        )
    else:
        assert all(
            any(
                call.args[1] == clf_name and
                call.args[5] == norm_input
                for call in mock_run.call_args_list
            )
            for clf_name in clfs
            for norm_input in norm_methods
        )

@pytest.mark.parametrize("clf_input, current_dataset, norm_input, eval_method_input", 
                        [
                            ("artificial neural networks", "blobs", "unnormalized", "holdout"),
                            ("decision tree", "varied", "zscore", "random subsampling"),
                            ("support vector machine", "noisy circles", "unnormalized", "kfold"),
                            ("naive bayes", "noisy moons", "minmax", "leave-one-out")
                        ]
                        )
def test_classify_input_one_classifier_one_normalization(current_dataset, clf_input, norm_input, eval_method_input):
    original_X, y = datasets_dict[current_dataset]
    classify_input(original_X, y, current_dataset, clf_input, norm_input, eval_method_input)

@pytest.mark.parametrize("clf_input, current_dataset, eval_method_input", 
                        [
                            ("artificial neural networks", "blobs", "holdout"),
                            ("decision tree", "varied", "random subsampling"),
                            ("support vector machine", "noisy circles", "kfold"),
                            ("naive bayes", "noisy moons", "leave-one-out")
                        ]
                        )
def test_classify_input_one_classifier_all_normalization(current_dataset, clf_input, eval_method_input, mocker):
    norm_input = "all"
    original_X, y = datasets_dict[current_dataset]
    mock_run = mocker.patch("classifiers.run_classifier")
    norm_methods = list(normalization_methods)
    classify_input(original_X, y, current_dataset, clf_input, norm_input, eval_method_input)

    norm_methods = list(normalization_methods)

    assert all(
        any(
            call.args[5] == norm_input
            for call in mock_run.call_args_list
        )
        for norm_input in norm_methods
    )

@pytest.mark.parametrize("current_dataset, norm_input, eval_method_input",
                        [
                            ("blobs", "unnormalized", "holdout"),
                            ("anisotropic", "minmax", "random subsampling"),
                            ("varied", "zscore", "kfold"),
                            ("noisy moons", "unnormalized", "leave-one-out")
                        ]
                        )
def test_classify_input_all_classifiers_one_normalization(current_dataset, norm_input, eval_method_input, mocker):
    mock_run = mocker.patch("classifiers.run_classifier")
    clf_input = "all"
    original_X, y = datasets_dict[current_dataset]
    classify_input(original_X, y, current_dataset, clf_input, norm_input, eval_method_input)

    clfs = list(main_classifiers)
    clfs_no_ann = [clf for clf in clfs if clf != "artificial neural networks"]

    if eval_method_input == "leave-one-out":
        assert all(
            any(
                call.args[1] == clf_name
                for call in mock_run.call_args_list
            )
            for clf_name in clfs_no_ann
        )

    else:
        assert all(
            any(
                call.args[1] == clf_name
                for call in mock_run.call_args_list
            )
            for clf_name in clfs
        )

def test_classify_input_loo_ann():
    current_dataset = "blobs"
    original_X, y = datasets_dict[current_dataset]
    clf_input = "artificial neural networks"
    norm_input = "unnormalized"
    eval_method_input = "leave-one-out"
    with pytest.raises(ValueError):
        classify_input(original_X, y, current_dataset, clf_input, norm_input, eval_method_input)