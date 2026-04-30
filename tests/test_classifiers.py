from classifiers import get_x_input, get_best, get_clf_input, classify_holdout, classify_knn_holdout, classify_input, main_classifiers, normalization_methods, classify_random_subsampling, main_classifiers, normalization_methods, classify_knn_random_subsampling, get_evaluation_method, evaluation_methods
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import pytest

@pytest.mark.parametrize("user_input", ["unnormalized", "minmax", "zscore"])
def test_get_x_input_valid(monkeypatch, user_input):
    monkeypatch.setattr("builtins.input", lambda _: user_input)
    result = get_x_input()
    assert result == user_input

def test_get_x_input_invalid_then_valid(monkeypatch, capsys):
    inputs = iter(["invalid", "unnormalized"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    result = get_x_input()
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
    result = get_best(test_dict)
    assert result == "Lucy"

@pytest.mark.parametrize("user_input", ["naive bayes", "decision tree", "support vector machine", "artificial neural networks", "k-nearest-neighbor"])
def test_get_clf_input_valid(monkeypatch, user_input):
    monkeypatch.setattr("builtins.input", lambda _: user_input)
    result = get_clf_input()
    assert result == user_input

def test_get_clf_input_invalid_then_valid(monkeypatch, capsys):
    inputs = iter(["invalid", "decision tree"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    result = get_clf_input()
    captured = capsys.readouterr()
    assert "Invalid input" in captured.out
    assert result == "decision tree"

@pytest.mark.parametrize("user_input", ["nAIVE BAYES", "DECISion TReE", "SUppORt VecToR mAchInE", "ArtIfICial neURAL NETWORKS", "K-NEAREST-NEIGHBOR"])
def test_get_clf_input_valid_case_insensitive(monkeypatch, user_input):
    monkeypatch.setattr("builtins.input", lambda _: user_input)
    result = get_clf_input()
    assert result == user_input.lower()

def test_classify_holdout(capsys):
    X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8)
    clf = GaussianNB()
    clf_name = "naive bayes"
    current_dataset = "blobs"
    normalization_method = "unnormalized"
    classify_holdout(clf, clf_name, X, y, current_dataset, normalization_method)
    captured = capsys.readouterr()
    assert current_dataset in captured.out
    assert clf_name in captured.out
    assert normalization_method in captured.out

def test_classify_knn_holdout(capsys):
    X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8)
    clf_name = "k-nearest-neighbor"
    current_dataset = "blobs"
    normalization_method = "unnormalized"
    classify_knn_holdout(clf_name, X, y, current_dataset, normalization_method)
    k_vals = [3, 5, 7]
    captured = capsys.readouterr()
    lines = captured.out.strip().split('\n')
    assert current_dataset in captured.out
    assert clf_name in captured.out
    assert normalization_method in captured.out

    for k in k_vals:
        assert any(str(k) in line and "k-nearest-neighbor" in line for line in lines)

@pytest.mark.parametrize("clf_name, current_dataset, normalization_method", 
                       [
                           ("naive bayes", "noisy_circles", "unnormalized"),
                           ("decision tree", "noisy_moons", "minmax"),
                           ("support vector machine", "blobs", "zscore"),
                           ("artificial neural networks", "varied", "unnormalized"),
                           ("naive bayes", "varied", "minmax")
                       ]
                       )
def test_classify_random_subsampling(capsys, clf_name, current_dataset, normalization_method):
    X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8)
    classify_random_subsampling(main_classifiers[clf_name], clf_name, X, y, current_dataset, normalization_method)
    captured = capsys.readouterr()
    assert current_dataset in captured.out
    assert clf_name in captured.out
    assert normalization_method in captured.out

@pytest.mark.parametrize("current_dataset, normalization_method", 
                        [
                            ("noisy_circles", "unnormalized"),
                            ("noisy_moons", "minmax"),
                            ("blobs", "zscore"),
                            ("anisotropic", "unnormalized"),
                            ("varied", "minmax")
                        ]
                        )
def test_classify_knn_random_subsampling(capsys, current_dataset, normalization_method):
    X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8)
    clf_name = "k-nearest-neighbor"
    classify_knn_random_subsampling(clf_name, X, y, current_dataset, normalization_method)
    k_vals = [3, 5, 7]
    captured = capsys.readouterr()
    lines = captured.out.strip().split('\n')

    assert current_dataset in captured.out
    assert clf_name in captured.out
    assert normalization_method in captured.out

    for k in k_vals:
        assert any(str(k) in line and "k-nearest-neighbor" in line for line in lines)

@pytest.mark.parametrize("user_input",
    list(evaluation_methods.keys())
)
def test_get_evaluation_method_valid(monkeypatch, user_input):
    monkeypatch.setattr("builtins.input", lambda _: user_input)
    result = get_evaluation_method()
    assert result == user_input.capitalize()

@pytest.mark.parametrize("user_input",
    [m.upper() for m in evaluation_methods.keys()]
)
def test_get_evaluation_method_case_insensitive(monkeypatch, user_input):
    monkeypatch.setattr("builtins.input", lambda _: user_input)
    result = get_evaluation_method()
    assert result == user_input.capitalize()

def test_get_evaluation_method_invalid_valid(monkeypatch, capsys):
    inputs = iter(["invalid", list(evaluation_methods.keys())[0]])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    result = get_evaluation_method()
    captured = capsys.readouterr()
    assert "Invalid input" in captured.out
    assert result == list(evaluation_methods.keys())[0]

# TODO: Add more tests for the classify_input function
def test_classify_input_holdout_all_classifiers_all_normalization(capsys):
    original_X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8)
    current_dataset = "blobs"
    k_vals = [3, 5, 7]
    classify_input(original_X, y, current_dataset, "all", "all", "Holdout")
    captured = capsys.readouterr()
    lines = captured.out.strip().split('\n')

    clfs = list(main_classifiers)
    normalizations = list(normalization_methods)

    for clf_name in clfs:
        for method in normalizations:
            assert any(clf_name in line and method in line for line in lines)

    for k in k_vals:
        assert any(str(k) in line and "k-nearest-neighbor" in line for line in lines)

    assert any("Best k-value" in line for line in lines)
