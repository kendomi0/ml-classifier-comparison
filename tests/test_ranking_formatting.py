import pytest

from ml_comparison.classifiers import ClassificationResult

from ml_comparison.ranking_formatting import (
    display_combinations_heading,
    combination_headings,
    get_results_with_same_scores,
    format_percent,
    get_decimal_places,
    rank_results,
    get_table_headers,
    format_results,
    get_given_results,
    get_given_ranked_results,
    FormattedResult
)

results_unique_scores = [
    ClassificationResult(
        classifier_name="naive bayes",
        evaluation_method="holdout",
        normalization_method="minmax",
        dataset="blobs",
        score=97.18,
        cost_tier=1,
        runtime=0.142,
    ),
    ClassificationResult(
        classifier_name="naive bayes",
        evaluation_method="holdout",
        normalization_method="unnormalized",
        dataset="blobs",
        score=95.58,
        cost_tier=1,
        runtime=0.132,
    ),
    ClassificationResult(
        classifier_name="naive bayes",
        evaluation_method="kfold",
        normalization_method="zscore",
        dataset="blobs",
        score=96.23,
        cost_tier=2,
        runtime=0.481,
    ),
    ClassificationResult(
        classifier_name="naive bayes",
        evaluation_method="random subsampling",
        normalization_method="minmax",
        dataset="blobs",
        score=91.37,
        cost_tier=2,
        runtime=0.517,
    ),
    ClassificationResult(
        classifier_name="naive bayes",
        evaluation_method="random subsampling",
        normalization_method="unnormalized",
        dataset="blobs",
        score=100.00,
        cost_tier=2,
        runtime=0.123,
    ),
]

results_recurring_scores = [
    ClassificationResult(
        classifier_name="naive bayes",
        evaluation_method="holdout",
        normalization_method="minmax",
        dataset="blobs",
        score=97.19,
        runtime=0.4926,
    ),
    ClassificationResult(
        classifier_name="artificial neural networks",
        evaluation_method="random subsampling",
        normalization_method="zscore",
        dataset="blobs",
        score=94.12,
        runtime=0.75846,
    ),
    ClassificationResult(
        classifier_name="naive bayes",
        evaluation_method="holdout",
        normalization_method="unnormalized",
        dataset="blobs",
        score=94.12,
        runtime=0.27153,
    ),
    ClassificationResult(
        classifier_name="naive bayes",
        evaluation_method="kfold",
        normalization_method="zscore",
        dataset="blobs",
        score=94.12,
        runtime=0.27178,
    ),
    ClassificationResult(
        classifier_name="naive bayes",
        evaluation_method="random subsampling",
        normalization_method="unnormalized",
        dataset="blobs",
        score=95.8,
        runtime=0.1235
    ),
]


def test_convert_percentage():
    score = 73.5
    converted_percentage = format_percent(score)
    assert converted_percentage == "73.50%"


def test_rank_results_by_runtime():
    sorted_results = rank_results(results_recurring_scores)
    assert sorted_results == [
        results_recurring_scores[0],
        results_recurring_scores[4],
        results_recurring_scores[2],
        results_recurring_scores[3],
        results_recurring_scores[1],
    ]


def test_display_combinations_heading_some():
    heading = display_combinations_heading(results_unique_scores, 2)
    assert heading == combination_headings["some"].format(
        number_of_combos=2, total_results=len(results_unique_scores)
    )


def test_display_combinations_heading_all():
    heading = display_combinations_heading(results_unique_scores, 5)
    assert heading == combination_headings["all"].format(
        number_of_combos=5, total_results=len(results_unique_scores)
    )


def test_display_combinations_heading_best():
    heading = display_combinations_heading(results_unique_scores, 1)
    assert heading == combination_headings["best"].format(
        number_of_combos=5, total_results=len(results_unique_scores)
    )


def test_display_combinations_heading_only():
    results = results_unique_scores[:1]
    heading = display_combinations_heading(results, 1)
    assert heading == combination_headings["only"]


def test_get_results_with_same_scores():
    results = get_results_with_same_scores(results_recurring_scores)
    expected_results = {
        94.12: [
            results_recurring_scores[1],
            results_recurring_scores[2],
            results_recurring_scores[3],
        ]
    }
    assert results == expected_results


def test_get_decimal_places():
    decimal_places = get_decimal_places(
        result=results_recurring_scores[2],
        tied_results=[
            results_recurring_scores[1],
            results_recurring_scores[2],
            results_recurring_scores[3],
        ],
        decimals=3,
    )
    assert decimal_places == 4


def test_rank_results_with_unique_scores():
    ranked_results = rank_results(results_unique_scores)
    assert ranked_results == [
        results_unique_scores[4],
        results_unique_scores[0],
        results_unique_scores[2],
        results_unique_scores[1],
        results_unique_scores[3],
    ]


@pytest.mark.parametrize(
    "number_of_combinations, expected_headers", 
    [
        (3, ["Ranking", "Classifier", "Normalization Method", "Accuracy", "Runtime (s)", "Computational Cost Tier"]),
        (1, ["Dataset", "Evaluation Method", "Classifier", "Normalization Method", "Accuracy", "Runtime (s)", "Computational Cost Tier"])
    ]

)
def test_get_table_headers(number_of_combinations, expected_headers):
    headers = get_table_headers(number_of_combinations)
    assert headers == expected_headers


def test_format_results():
    results = [
        results_recurring_scores[0],
        results_recurring_scores[2],
        results_recurring_scores[3]
    ]
    formatted_results = format_results(results, 3)
    assert formatted_results == [
        FormattedResult(
            dataset="Blobs",
            evaluation_method="Holdout",
            classifier="Naive Bayes",
            normalization_method="Minmax",
            formatted_score="97.19%",
            cost_tier=1,
            formatted_runtime="0.493",
            ranking=1
        ),
        FormattedResult(
            dataset="Blobs",
            evaluation_method="Holdout",
            classifier="Naive Bayes",
            normalization_method="Unnormalized",
            formatted_score="94.12%",
            cost_tier=1,
            formatted_runtime="0.2715",
            ranking=2
        ),
        FormattedResult(
            dataset="Blobs",
            evaluation_method="Kfold",
            classifier="Naive Bayes",
            normalization_method="Zscore",
            formatted_score="94.12%",
            cost_tier=2,
            formatted_runtime="0.2718",
            ranking=3
        )
    ]

def test_get_given_results():
    given_results = get_given_results(results_unique_scores, 2)
    assert given_results == [
        results_unique_scores[0],
        results_unique_scores[1],
    ]

def test_get_given_ranked_results():
    results = get_given_ranked_results(results_unique_scores, 3)
    assert results == [
        results_unique_scores[4],
        results_unique_scores[0],
        results_unique_scores[2],
    ]