from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml_comparison.classifiers import ClassificationResult


combination_headings = {
    "some": "Top {number_of_combos} out of {total_results} total combinations: ",
    "all": "All combinations: ",
    "best": "Best combination: ",
    "only": "Combination: ",
}


def format_percent(percent):
    return f"{percent:.2f}%"


def rank_results(results):
    ranked_results = sorted(results, key=lambda result: (-result.score, result.runtime))
    return ranked_results


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


def format_combination(results):
    combos = []
    results_with_duplicate_scores = get_results_with_same_scores(results)
    for index, result in enumerate(results):
        decimal_places = 3
        if result.score in results_with_duplicate_scores:
            decimal_places = get_decimal_places(
                result=result,
                tied_results=results_with_duplicate_scores[result.score],
                decimals=decimal_places,
            )
        formatted_score = format_percent(result.score)
        print_kfold_value = ""
        if result.kfold_value is not None:
            print_kfold_value = f" (Kfold Value: {result.kfold_value})"
        print_knn_value = ""
        if result.knn_value is not None:
            print_knn_value = f" (KNN Value: {result.knn_value})"
        display = f"{result.dataset.title()} ⋅ {result.classifier_name.title()}{print_knn_value} ⋅ {result.evaluation_method.title()}{print_kfold_value} ⋅ {result.normalization_method.title()} |  Accuracy: {formatted_score} ⋅ Runtime: {result.runtime:.{decimal_places}f} seconds ⋅ Computational Cost Tier: {result.cost_tier}"
        if len(results) > 1:
            display = f"#{index+1}: {display}"
        combos.append(display)
    return combos


def display_selected_combos(results, number_of_combos):
    heading = display_combinations_heading(results, number_of_combos)
    combinations = []
    if len(results) == 1:
        combinations = format_combination(results)
    else:
        ranked_results = rank_results(results)
        if number_of_combos == 1:
            combinations = format_combination(ranked_results[:1])
        else:
            combinations = format_combination(ranked_results[:number_of_combos])
    return heading, combinations
