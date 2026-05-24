from plotting import create_scatter_plot, trigger_plot_creation
from sklearn import datasets

def test_create_scatter_plot(mocker):
    mock_plt = mocker.patch("plotting.plt")

    X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8)
    
    current_dataset = "noisy circles"

    create_scatter_plot(X, y, current_dataset)

    mock_plt.figure.assert_called_once()
    mock_plt.scatter.assert_called_once()
    mock_plt.title.assert_called_once_with(f"{current_dataset.capitalize()} dataset")
    mock_plt.show.assert_called_once()

def test_trigger_plot_creation_yes(mocker, monkeypatch, capsys):
    mock_create_scatter_plot = mocker.patch("plotting.create_scatter_plot")
    X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8)
    current_dataset = "noisy circles"
    inputs = iter(["l", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    trigger_plot_creation(X, y, current_dataset)
    captured = capsys.readouterr()
    assert "Invalid input" in captured.out
    mock_create_scatter_plot.assert_called_once()

def test_trigger_plot_creation_no(mocker, monkeypatch, capsys):
    mock_create_scatter_plot = mocker.patch("plotting.create_scatter_plot")
    X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8)
    current_dataset = "noisy circles"
    inputs = iter(["sksks", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    trigger_plot_creation(X, y, current_dataset)
    captured = capsys.readouterr()
    assert "Invalid input" in captured.out
    mock_create_scatter_plot.assert_not_called()