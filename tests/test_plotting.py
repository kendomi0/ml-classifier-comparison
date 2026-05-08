from plotting import create_scatter_plot
from sklearn import datasets
import matplotlib.pyplot as plt

def test_create_scatter_plot(mocker):
    mock_plt = mocker.patch("plotting.plt")

    X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8)
    
    current_dataset = "noisy circles"

    create_scatter_plot(X, y, current_dataset)

    mock_plt.figure.assert_called_once()
    mock_plt.scatter.assert_called_once()
    mock_plt.title.assert_called_once_with(f"{current_dataset.capitalize()} dataset")
    mock_plt.show.assert_called_once()
