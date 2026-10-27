import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

TESTED_K_VALUES = [3, 4, 5]
CLUSTER_NUMBER = 4
CLUSTER_LABELS = list(range(1, CLUSTER_NUMBER + 1))

RANDOM_STATE = 0
N_INIT = 10

BASE_DIR = "ml_comparison/cost_calculations"

def load_median_runtimes():
    raw_results = pd.read_csv(f"{BASE_DIR}/raw_results.csv")
    medians = raw_results.groupby([
        "dataset", 
        "classifier", 
        "evaluation_method", 
        "normalization_method"])["runtime"].median().reset_index(name="median_time")
    medians["cost_tier"] = None
    return medians

def calculate_log_times(combinations):
    return np.log(combinations["median_time"].to_numpy())

def calculate_gaps(
        log_times,
        ):
    sorted_times = np.sort(log_times)
    unsorted_gaps = np.diff(sorted_times)
    sorted_gaps =  np.round(np.sort(unsorted_gaps)[::-1], 3).tolist()
    return sorted_gaps

def calculate_silhouette_scores(
        reshaped_times,
        k_values,
        ):
    k_scores = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(reshaped_times)
        score = silhouette_score(reshaped_times, km.labels_)
        k_scores.append(
            {"k": k, "silhouette": score}
        )
    return k_scores

def cluster_runtimes(reshaped_times):
    km = KMeans(n_clusters=CLUSTER_NUMBER,
                n_init=N_INIT,
                random_state=RANDOM_STATE
                ).fit(
        reshaped_times
        )
    return km

def label_clusters(km):
    ordered_clusters = np.argsort(km.cluster_centers_.ravel())
    names = dict(zip(ordered_clusters, CLUSTER_LABELS))
    labeled_clusters = [names[cluster] for cluster in km.labels_]
    return labeled_clusters

def analyze_runtimes(df):
    df = df.copy()
    silhouettes_list = []
    gap_dict = {}
    for dataset, combinations in df.groupby("dataset"):
        log_times = calculate_log_times(combinations)
        sorted_gaps = calculate_gaps(log_times)
        gap_dict[dataset] = sorted_gaps

        reshaped_times = log_times.reshape(-1, 1)
        silhouettes = calculate_silhouette_scores(
            reshaped_times,
            TESTED_K_VALUES,
            )
        for silhouette in silhouettes:
            silhouette["dataset"] = dataset
            silhouettes_list.append(silhouette)
        km = cluster_runtimes(reshaped_times)
        labeled_clusters = label_clusters(km)
        df.loc[combinations.index, "cost_tier"] = labeled_clusters
    return df, silhouettes_list, gap_dict

def get_tier_min_and_max(df):
    print(df)
    results = df.groupby(['dataset', 'cost_tier']).agg(
        min_median=('median_time', 'min'),
        max_median=('median_time', 'max')
    )
    return results

def save_results(df, scores, gaps_dict):
    df.to_csv(f"{BASE_DIR}/results_with_costs.csv", index=False)

    tier_mins_maxes = get_tier_min_and_max(df)
    tier_mins_maxes_df = pd.DataFrame(tier_mins_maxes)
    tier_mins_maxes_df.to_csv(f"{BASE_DIR}/tier_mins_maxes.csv")

    silhouette_scores_df = pd.DataFrame(scores)
    silhouette_scores_df = silhouette_scores_df[["dataset", "k", "silhouette"]]
    silhouette_scores_df.to_csv(f"{BASE_DIR}/silhouette_scores.csv",
                                index=False)

    with open(f"{BASE_DIR}/gaps.json", "w", encoding="utf-8") as file:
        json.dump(gaps_dict, file, indent=2)

if __name__ == "__main__":
    median_runtimes = load_median_runtimes()
    cost_results, silhouette_scores, gaps = analyze_runtimes(median_runtimes)
    save_results(cost_results, silhouette_scores, gaps)
