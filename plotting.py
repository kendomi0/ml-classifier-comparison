import matplotlib.pyplot as plt
from sklearn import datasets

def create_scatter_plot(X, y, current_dataset):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='plasma', edgecolors='k')
    plt.title(f"{current_dataset.capitalize()} dataset")
    plt.show()