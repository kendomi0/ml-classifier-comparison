from typing import TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from ml_comparison.classifiers import ClassificationResult

def get_cost_tier(result: ClassificationResult):
    result_costs = pd.read_csv("ml_comparison/cost_calculations/results_with_costs.csv")
    return result_costs.loc[
        (result_costs["dataset"] == result.dataset)
        & (result_costs["classifier"] == result.classifier_name)
        & (result_costs["evaluation_method"] == result.evaluation_method)
        & (result_costs["normalization_method"] == result.normalization_method),
        "cost_tier",
    ].values[0]