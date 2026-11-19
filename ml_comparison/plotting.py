from ml_comparison.data import datasets_dict
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
output_dir = BASE_DIR / "static" / "images" / "plots"

def create_and_save_scatter_plot(X, y, current_dataset, use_dark):
    if use_dark:
        theme = "dark"
        style = "dark_background"
        color_map = "Purples_r"
    else:
        theme = "light"
        style = "default"
        color_map = "Purples"
    file_path = f"{current_dataset.replace(" ", "_")}_plot_{theme}.png"
    with plt.style.context(style):
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=color_map, edgecolors='k')
        plt.title(f"{current_dataset.title()}")
        plt.savefig(output_dir / file_path, dpi=300, bbox_inches="tight")


def generate_plots(datasets):
    for dataset_name, data in datasets.items():
        for value in [True, False]:
            X, y = data
            create_and_save_scatter_plot(
                X=X,
                y=y,
                current_dataset=dataset_name,
                use_dark=value
            )