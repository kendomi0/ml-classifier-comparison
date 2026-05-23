from classifiers import get_normalization_method, get_best, get_classifier, classify_holdout, classify_holdout_knn, run_classifier, classifier_map, normalization_methods, classify_random_subsampling, classifier_map, normalization_methods, classify_random_subsampling_knn, get_evaluation_method, evaluation_methods, classify, classify_split, classify_kfold, classify_kfold_knn, classify_loo, classify_loo_knn, ClassificationResult, sort_by_cost, print_combination, display_selected_combos, calculate_cost, get_valid_number_of_combos, validate_combo_number_input, rank_results, prompt_number_of_combinations, check_valid_combination
from sklearn.model_selection import KFold, LeaveOneOut
from data import datasets_dict
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import pytest
from unittest.mock import MagicMock


@pytest.mark.parametrize("user_input", ["unnormalized", "minmax", "zscore"])
def test_get_normalization_method_valid(monkeypatch, user_input):
    monkeypatch.setattr("builtins.input", lambda _: user_input)
    result = get_normalization_method()
    assert result == user_input

def test_get_normalization_method_invalid_then_valid(monkeypatch, capsys):
    inputs = iter(["invalid", "unnormalized"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    result = get_normalization_method()
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
def test_get_classifier_valid(monkeypatch, user_input):
    monkeypatch.setattr("builtins.input", lambda _: user_input)
    evaluation_method="holdout"
    result = get_classifier(evaluation_method)
    assert result == user_input

def test_get_classifier_invalid_then_valid(monkeypatch, capsys):
    inputs = iter(["invalid", "decision tree"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    evaluation_method="holdout"
    result = get_classifier(evaluation_method)
    captured = capsys.readouterr()
    assert "Invalid input" in captured.out
    assert result == "decision tree"

@pytest.mark.parametrize("user_input", ["nAIVE BAYES", "DECISion TReE", "SUppORt VecToR mAchInE", "ArtIfICial neURAL NETWORKS", "K-NEAREST-NEIGHBOR"])
def test_get_classifier_valid_case_insensitive(monkeypatch, user_input):
    monkeypatch.setattr("builtins.input", lambda _: user_input)
    evaluation_method="holdout"
    result = get_classifier(evaluation_method)
    assert result == user_input.lower()

def test_get_classifier_loo_ann(monkeypatch):
    inputs = iter(["artificial neural networks", "naive bayes"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    evaluation_method = "leave-one-out"
    get_classifier(evaluation_method)

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

def test_classify_holdout():
    X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8)
    clf = GaussianNB()
    clf_name = "naive bayes"
    current_dataset = "blobs"
    normalization_method = "unnormalized"
    result = classify_holdout(clf, clf_name, X, y, current_dataset, normalization_method)
    print(result)
    assert isinstance(result, ClassificationResult)

def test_classify_holdout_knn():
    X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8)
    clf_name = "k-nearest-neighbor"
    current_dataset = "blobs"
    normalization_method = "unnormalized"
    result = classify_holdout_knn(clf_name, X, y, current_dataset, normalization_method)
    assert isinstance(result, ClassificationResult)


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
    result = classify_random_subsampling(clf, clf_name, X, y, current_dataset, normalization_method)
    assert isinstance(result, ClassificationResult)

@pytest.mark.parametrize("current_dataset, normalization_method",
                        [
                            ("noisy circles", "unnormalized"),
                            ("noisy moons", "minmax"),
                            ("blobs", "zscore"),
                            ("anisotropic", "unnormalized"),
                            ("varied", "minmax")
                        ]
                        )
def test_classify_random_subsampling_knn(current_dataset, normalization_method):
    X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8)
    clf_name = "k-nearest-neighbor"
    result = classify_random_subsampling_knn(clf_name, X, y, current_dataset, normalization_method)
    assert isinstance(result, ClassificationResult)

def test_classify_split():
    clf = classifier_map["naive bayes"]
    current_dataset = "blobs"
    X, y = datasets_dict[current_dataset]
    splitter = KFold(n_splits=3, shuffle=True)
    result = classify_split(clf, splitter, X, y)
    assert isinstance(result, list)
    for score in result:
        assert 0 < score <= 1.0

def test_classify_kfold():
    clf_name = "naive bayes"
    clf = classifier_map[clf_name]
    current_dataset = "blobs"
    normalization_method = "unnormalized"
    X, y = datasets_dict[current_dataset]
    result = classify_kfold(clf, clf_name, X, y, current_dataset, normalization_method)
    assert isinstance(result, ClassificationResult)

def test_classify_kfold_knn():
    classifier_name = "k-nearest-neighbor"
    current_dataset = "blobs"
    normalization_method = "minmax"
    X, y = datasets_dict[current_dataset]
    result = classify_kfold_knn(classifier_name, X, y, current_dataset, normalization_method)
    assert isinstance(result, ClassificationResult)

def test_classify_loo():
    current_dataset = "blobs"
    X, y = datasets_dict[current_dataset]
    clf_name = "naive bayes"
    clf = classifier_map[clf_name]
    normalization_method = "unnormalized"
    result = classify_loo(clf, clf_name, X, y, current_dataset, normalization_method)
    assert isinstance(result, ClassificationResult)

def test_classify_loo_knn(mocker):
    mock_split = mocker.patch("classifiers.classify_split")
    current_dataset = "blobs"
    X, y = datasets_dict[current_dataset]
    normalization_method = "unnormalized"
    result = classify_loo_knn("k-nearest-neighbor", X, y, current_dataset, normalization_method)
    assert isinstance(result, ClassificationResult)
    assert any(
        isinstance(call.args[1], LeaveOneOut) for call in mock_split.call_args_list
        )
    
@pytest.mark.parametrize("clf_name, current_dataset, normalization_method, evaluation_method",
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
def test_classify(clf_name, current_dataset, normalization_method, evaluation_method, mocker):
    mock_dict = mocker.patch("classifiers.evaluation_methods")
    clf = classifier_map[clf_name]
    X, y = datasets_dict[current_dataset]
    result_lst = []
    classify(clf, clf_name, X, y, current_dataset, normalization_method, evaluation_method, result_lst)
    if clf_name != "k-nearest-neighbor":
        mock_dict[evaluation_method][0].assert_called_once()
    else:
        mock_dict[evaluation_method][1].assert_called_once()

def test_check_valid_combination_both(monkeypatch, capsys):
    inputs = iter(["ajjj", "both", "leave-one-out", "all"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    evaluation_method, classifier_name = check_valid_combination("leave-one-out", "artificial neural networks")
    captured = capsys.readouterr()
    assert "Invalid input" in captured.out
    assert evaluation_method == "leave-one-out"
    assert classifier_name == "all"

def test_check_valid_combination_clf(monkeypatch, capsys):
    inputs = iter(["ajjj", "classifier", "artificial neural networks", "naive bayes"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    evaluation_method, classifier_name = check_valid_combination("leave-one-out", "artificial neural networks")
    captured = capsys.readouterr()
    assert "Invalid input" in captured.out
    assert evaluation_method == "leave-one-out"
    assert classifier_name == "naive bayes"

def test_check_valid_combination_evaluation(monkeypatch, capsys):
    inputs = iter(["ajjj", "evaluation method", "leave-one-out", "classifier", "all"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    evaluation_method, classifier_name = check_valid_combination("leave-one-out", "artificial neural networks")
    captured = capsys.readouterr()
    assert "Invalid input" in captured.out
    assert evaluation_method == "leave-one-out"
    assert classifier_name == "all"

def test_check_valid_combination_already_valid():
    evaluation_method, classifier_name = check_valid_combination("leave-one-out", "naive bayes")
    assert evaluation_method == "leave-one-out"
    assert classifier_name == "naive bayes"

def test_run_classifier_loo_ann(mocker):
    current_dataset = "blobs"
    X, y = datasets_dict[current_dataset]
    mocker.patch("classifiers.classify")
    mock_check_valid_combination = mocker.patch("classifiers.check_valid_combination", return_value=("all", "all"))
    run_classifier(X, y, current_dataset, evaluation_method="leave-one-out", normalization_method="all", classifier_name="artificial neural networks")
    mock_check_valid_combination.assert_called_once()
        
def test_run_classifier_all_evaluations_normalizations_clfs(mocker):
    current_dataset = "blobs"
    X, y = datasets_dict[current_dataset]
    mock_run = mocker.patch("classifiers.classify")
    run_classifier(X, y, current_dataset, "all", "all", "all")
    clfs = list(classifier_map)
    norm_methods = list(normalization_methods)
    evaluation_methods_list = list(evaluation_methods)

    assert len(mock_run.call_args_list) == 57

    assert all(
        any(
            call.args[1] == clf_name and
            call.args[5] == normalization_method and
            call.args[6] == evaluation_method
            for call in mock_run.call_args_list
        )
        for clf_name in clfs
        for normalization_method in norm_methods
        for evaluation_method in evaluation_methods_list
        if not (evaluation_method == "leave-one-out" and clf_name == "artificial neural networks")
    )

@pytest.mark.parametrize("evaluation_method",
                         ["holdout", "random subsampling", "kfold", "leave-one-out"]
                         )
def test_run_classifier_all_classifiers_all_normalization(evaluation_method, mocker):
    current_dataset = "blobs"
    X, y = datasets_dict[current_dataset]
    mock_run = mocker.patch("classifiers.classify")
    run_classifier(X, y, current_dataset, "all", "all", evaluation_method)

    clfs = list(classifier_map)
    clfs_no_ann = [clf for clf in clfs if clf != "artificial neural networks"]
    norm_methods = list(normalization_methods)

    if evaluation_method == "leave-one-out":
        assert all(
            any(
                call.args[1] == clf_name and
                call.args[5] == normalization_method
                for call in mock_run.call_args_list
            )
            for clf_name in clfs_no_ann
            for normalization_method in norm_methods
        )
    else:
        assert all(
            any(
                call.args[1] == clf_name and
                call.args[5] == normalization_method
                for call in mock_run.call_args_list
            )
            for clf_name in clfs
            for normalization_method in norm_methods
        )

@pytest.mark.parametrize("classifier, current_dataset, normalization_method, evaluation_method", 
                        [
                            ("artificial neural networks", "blobs", "unnormalized", "holdout"),
                            ("decision tree", "varied", "zscore", "random subsampling"),
                            ("support vector machine", "noisy circles", "unnormalized", "kfold"),
                            ("naive bayes", "noisy moons", "minmax", "leave-one-out")
                        ]
                        )
def test_run_classifier_one_classifier_one_normalization(current_dataset, classifier, normalization_method, evaluation_method):
    original_X, y = datasets_dict[current_dataset]
    result_lst = run_classifier(original_X, y, current_dataset, classifier, normalization_method, evaluation_method)
    assert len(result_lst) == 1
    assert all(
        isinstance(result, ClassificationResult) for result in result_lst
    )

@pytest.mark.parametrize("classifier, current_dataset, evaluation_method",
                        [
                            ("artificial neural networks", "blobs", "holdout"),
                            ("decision tree", "varied", "random subsampling"),
                            ("support vector machine", "noisy circles", "kfold"),
                            ("naive bayes", "noisy moons", "leave-one-out")
                        ]
                        )
def test_run_classifier_one_classifier_all_normalization(current_dataset, classifier, evaluation_method, mocker):
    normalization_method = "all"
    original_X, y = datasets_dict[current_dataset]
    mock_run = mocker.patch("classifiers.classify")
    norm_methods = list(normalization_methods)
    run_classifier(original_X, y, current_dataset, classifier, normalization_method, evaluation_method)

    norm_methods = list(normalization_methods)

    assert all(
        any(
            call.args[5] == normalization_method
            for call in mock_run.call_args_list
        )
        for normalization_method in norm_methods
    )

@pytest.mark.parametrize("current_dataset, normalization_method, evaluation_method",
                        [
                            ("blobs", "unnormalized", "holdout"),
                            ("anisotropic", "minmax", "random subsampling"),
                            ("varied", "zscore", "kfold"),
                            ("noisy moons", "unnormalized", "leave-one-out")
                        ]
                        )
def test_run_classifier_all_classifiers_one_normalization(current_dataset, normalization_method, evaluation_method, mocker):
    mock_run = mocker.patch("classifiers.classify")
    classifier = "all"
    original_X, y = datasets_dict[current_dataset]
    run_classifier(original_X, y, current_dataset, classifier, normalization_method, evaluation_method)

    clfs = list(classifier_map)
    clfs_no_ann = [clf for clf in clfs if clf != "artificial neural networks"]

    if evaluation_method == "leave-one-out":
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

def test_calculate_cost():
    list_classifier_map = list(classifier_map)
    list_evaluation_methods = list(evaluation_methods)
    list_normalization_methods = list(normalization_methods)
    list_datasets_dict = list(datasets_dict)

    result = ClassificationResult(
            classifier_name=list_classifier_map[0],
            evaluation_method=list_evaluation_methods[0],
            normalization_method=list_normalization_methods[0],
            dataset=list_datasets_dict[0],
            score="95.00%",
        )
    cost_of_result = calculate_cost(result)
    assert cost_of_result == 2

def test_sort_by_cost():
    results = [
        ClassificationResult(
            classifier_name="naive bayes",
            evaluation_method="holdout",
            normalization_method="minmax",
            dataset="blobs",
            score="97.00%",
        ),
        ClassificationResult(
            classifier_name="naive bayes",
            evaluation_method="holdout",
            normalization_method="unnormalized",
            dataset="blobs",
            score="94.00%",
        ),
        ClassificationResult(
            classifier_name="naive bayes",
            evaluation_method="kfold",
            normalization_method="zscore",
            dataset="blobs",
            score="94.00%",
        ),
        ClassificationResult(
            classifier_name="naive bayes",
            evaluation_method="random subsampling",
            normalization_method="unnormalized",
            dataset="blobs",
            score="95.00%",
        ),
    ]
    results_with_costs = results.copy()
    results_with_costs[1].cost = 2
    results_with_costs[2].cost = 5
    sorted_results = sort_by_cost(results)
    assert sorted_results == [results_with_costs[0], results_with_costs[3], results_with_costs[1], results_with_costs[2]]

def test_print_combination_isbest():
    result_list = [
        ClassificationResult(
            classifier_name="k-nearest-neighbor",
            evaluation_method="kfold",
            normalization_method="minmax",
            dataset="blobs",
            score="95.00%",
            knn_value=3,
            kfold_value=3
        )
    ]
    best_combo = print_combination(result_list, True)
    best_result = result_list[0]
    assert best_combo[0] == f"Best combo: {best_result.dataset.capitalize()}, {best_result.classifier_name.capitalize()} (KNN value: {best_result.knn_value}), {best_result.evaluation_method.capitalize()} (Kfold value: {best_result.kfold_value}), {best_result.normalization_method.capitalize()}. Accuracy: {best_result.score}"

@pytest.mark.parametrize("number_input", ["all", "4"])
def test_prompt_number_of_combinations(monkeypatch, number_input):
    results = [1, 2, 3, 4]
    monkeypatch.setattr("builtins.input", lambda _: number_input)
    number_of_combinations_input = prompt_number_of_combinations(results)
    assert number_of_combinations_input == 4

def test_validate_combo_number_input_invalid_valid(monkeypatch, capsys):
    results = ["abc", "def", "ghi"]
    inputs = iter(["4", "0", "3"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    result = validate_combo_number_input("ajaj", results)
    captured = capsys.readouterr()
    assert "Invalid string input" in captured.out
    assert "Number of combinations to display cannot be higher than number of total combinations" in captured.out
    assert "Number of combinations to display cannot be less than 1" in captured.out
    assert result == 3

def test_get_valid_number_of_combos(mocker):
    results = [
        ClassificationResult(
            classifier_name="naive bayes",
            evaluation_method="holdout",
            normalization_method="minmax",
            dataset="blobs",
            score="97.00%",
        ),
        ClassificationResult(
            classifier_name="naive bayes",
            evaluation_method="holdout",
            normalization_method="unnormalized",
            dataset="blobs",
            score="97.00%",
        ),
        ClassificationResult(
            classifier_name="naive bayes",
            evaluation_method="kfold",
            normalization_method="zscore",
            dataset="blobs",
            score="95.00%",
        )
    ]
        
    mock_prompt_number_of_combinations = mocker.patch("classifiers.prompt_number_of_combinations")
    mock_validate_combo_number_input = mocker.patch("classifiers.validate_combo_number_input")
    get_valid_number_of_combos(results)
    mock_prompt_number_of_combinations.assert_called()
    mock_validate_combo_number_input.assert_called_once()

def test_rank_results_with_recurring_scores(mocker):
    results = [
        ClassificationResult(
            classifier_name="naive bayes",
            evaluation_method="holdout",
            normalization_method="minmax",
            dataset="blobs",
            score="97.00%",
        ),
        ClassificationResult(
            classifier_name="naive bayes",
            evaluation_method="holdout",
            normalization_method="unnormalized",
            dataset="blobs",
            score="94.00%",
        ),
        ClassificationResult(
            classifier_name="naive bayes",
            evaluation_method="kfold",
            normalization_method="zscore",
            dataset="blobs",
            score="94.00%",
        ),
        ClassificationResult(
            classifier_name="naive bayes",
            evaluation_method="random subsampling",
            normalization_method="unnormalized",
            dataset="blobs",
            score="95.00%",
        ),
    ]
    mock_sort_by_cost = mocker.patch("classifiers.sort_by_cost")
    rank_results(results)
    mock_sort_by_cost.assert_called_once()

def test_rank_results_with_unique_scores():
    results = [
        ClassificationResult(
            classifier_name="naive bayes",
            evaluation_method="holdout",
            normalization_method="minmax",
            dataset="blobs",
            score="97.00%",
        ),
        ClassificationResult(
            classifier_name="naive bayes",
            evaluation_method="holdout",
            normalization_method="unnormalized",
            dataset="blobs",
            score="95.00%",
        ),
        ClassificationResult(
            classifier_name="naive bayes",
            evaluation_method="kfold",
            normalization_method="zscore",
            dataset="blobs",
            score="96.00%",
        ),
        ClassificationResult(
            classifier_name="naive bayes",
            evaluation_method="random subsampling",
            normalization_method="unnormalized",
            dataset="blobs",
            score="91.00%",
        ),
    ]
    ranked_results = rank_results(results)
    assert ranked_results == [results[0], results[2], results[1], results[3]]

def test_display_selected_combos(mocker):
    mock_print_combination = mocker.patch("classifiers.print_combination")
    mock_validate_combo_number_input = mocker.patch("classifiers.validate_combo_number_input")
    results = [
        ClassificationResult(
            classifier_name="naive bayes",
            evaluation_method="holdout",
            normalization_method="minmax",
            dataset="blobs",
            score="92.00%",
        ),
        ClassificationResult(
            classifier_name="naive bayes",
            evaluation_method="holdout",
            normalization_method="unnormalized",
            dataset="blobs",
            score="95.00%",
        ),
        ClassificationResult(
            classifier_name="naive bayes",
            evaluation_method="kfold",
            normalization_method="zscore",
            dataset="blobs",
            score="97.00%",
        )
    ]
    ranked_results = rank_results(results)
    display_selected_combos(results, 3)
    mock_print_combination.assert_called_once_with(ranked_results[:3], number_of_total_results=3)
