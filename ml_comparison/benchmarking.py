import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from ml_comparison.classifiers import run_classifier, evaluation_methods, classifier_map, normalization_methods
from ml_comparison.data import datasets_dict
import pandas as pd

all_results = []

for dataset_name, data in datasets_dict.items():
    for evaluation_method in evaluation_methods:
        for classifier_name in classifier_map:
            for normalization_method in normalization_methods:
                run = 0
                for i in range(10):
                    run += 1
                    X, y = data
                    results = run_classifier(
                        X,
                        y,
                        dataset_name,
                        classifier_name,
                        normalization_method,
                        evaluation_method,
                        )
                    for result in results:
                        result_dict = {
                            "dataset": result.dataset,
                            "classifier": result.classifier_name,
                            "evaluation_method": result.evaluation_method,
                            "normalization_method": result.normalization_method,
                            "runtime": result.runtime,
                            "run": run,
                        }
                        all_results.append(result_dict)
results_dataframe = pd.DataFrame(all_results)
results_dataframe.to_csv("ml_comparison/cost_calculations/raw_results.csv", index=False)

