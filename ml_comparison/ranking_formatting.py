from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass
from ml_comparison.cost_calculations.cost_tiers import get_cost_tier

if TYPE_CHECKING:
    from ml_comparison.classifiers import ClassificationResult

@dataclass
class FormattedResult:
    dataset: str
    evaluation_method: str
    classifier: str
    normalization_method: str
    formatted_score: str
    cost_tier: int
    formatted_runtime: str
    ranking: Optional[int] = None


combination_headings = {
    "some": "Top {number_of_combos} out of {total_results} total combinations: ",
    "all": "All combinations: ",
    "best": "Best combination: ",
    "only": "Combination: ",
}


def format_percent(percent):
    return f"{percent:.2f}%"


def get_given_results(results, number_of_combinations):
    return results[:number_of_combinations]


def rank_results(results):
    ranked_results = sorted(results, key=lambda result: (-result.score, result.runtime))
    return ranked_results


def get_given_ranked_results(results, number_of_combinations):
    ranked_results = rank_results(results)
    given_ranked_results = get_given_results(ranked_results, number_of_combinations)
    return given_ranked_results


def display_combinations_heading(results, number_of_combos):
    heading = ""
    total_results = len(results)
    if number_of_combos == 1:
        if total_results == 1:
            heading = combination_headings["only"]
        else:
            heading = combination_headings["best"]
    elif total_results > number_of_combos:
        heading = combination_headings["some"].format(
            number_of_combos=number_of_combos, total_results=total_results
        )
    else:
        heading = combination_headings["all"]
    return heading


def get_results_with_same_scores(results):
    scores = [result.score for result in results]
    duplicated_scores = [score for score in scores if scores.count(score) > 1]
    duplicated_scores = set(duplicated_scores)
    results_with_same_scores = {}
    for result in results:
        if result.score in duplicated_scores:
            results_with_same_scores.setdefault(result.score, []).append(result)
    return results_with_same_scores


def get_decimal_places(result: ClassificationResult, tied_results: list, decimals: int):
    result_runtime = result.runtime
    if result_runtime is None:
        return decimals
    runtimes = [round(tied_result.runtime, decimals) for tied_result in tied_results]
    while runtimes.count(round(result_runtime, decimals)) > 1:
        decimals += 1
        runtimes = [round(result.runtime, decimals) for result in tied_results]
    return decimals


def get_table_headers(number_of_combinations):
    headers = ["Dataset", "Evaluation Method", "Classifier", "Normalization Method", "Accuracy", "Runtime (s)", "Computational Cost Tier"]
    if number_of_combinations > 1:
        headers.insert(0, "Ranking")
    return headers

def format_results(results: list[ClassificationResult], number_of_combinations):
    results = get_given_ranked_results(results, number_of_combinations)
    formatted_results = []
    results_with_duplicate_scores = get_results_with_same_scores(results)
    for index, result in enumerate(results):
        decimal_places = 3
        if result.score in results_with_duplicate_scores:
            decimal_places = get_decimal_places(
                result=result,
                tied_results=results_with_duplicate_scores[result.score],
                decimals=decimal_places,
            )
        formatted_result = FormattedResult(
            dataset=result.dataset.title(),
            evaluation_method=result.evaluation_method.title(),
            classifier=result.classifier_name.title(),
            normalization_method=result.normalization_method.title(),
            formatted_score=format_percent(result.score),
            cost_tier=get_cost_tier(result), # type: ignore
            formatted_runtime=f"{result.runtime:.{decimal_places}f}",
            ranking=index+1
        )
        formatted_results.append(formatted_result)
    return formatted_results
