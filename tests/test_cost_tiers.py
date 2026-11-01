from ml_comparison.cost_calculations.cost_tiers import get_cost_tier
from ml_comparison.classifiers import ClassificationResult

def test_get_cost_tier():
    result = ClassificationResult(
        classifier_name="naive bayes",
        evaluation_method="random subsampling",
        normalization_method="unnormalized",
        dataset="blobs",
        score=100.00,
        cost_tier=2,
        runtime=0.123,
    )
    cost_tier = get_cost_tier(result)
    assert cost_tier == 2