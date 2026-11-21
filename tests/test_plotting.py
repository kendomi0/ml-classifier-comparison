import pytest

from ml_comparison.plotting import create_and_save_scatter_plot, generate_plots
from ml_comparison.data import datasets_dict


@pytest.mark.parametrize(
    "use_dark, style, cmap, file_name",
    [
        (True, "dark_background", "Purples_r", "noisy_circles_plot_dark.png"),
        (False, "default", "Purples", "noisy_circles_plot_light.png"),
    ],
)
def test_create_and_save_scatter_plot(mocker, use_dark, style, cmap, file_name):
    mock_plt = mocker.patch("ml_comparison.plotting.plt")
    dataset = "noisy circles"
    X, y = datasets_dict[dataset]

    create_and_save_scatter_plot(X, y, "noisy circles", use_dark)

    mock_plt.style.context.assert_called_once_with(style)
    mock_plt.scatter.assert_called_once()
    assert mock_plt.scatter.call_args.kwargs["cmap"] == cmap
    mock_plt.title.assert_called_once_with(dataset.title())
    assert mock_plt.savefig.call_args.args[0].name == file_name


def test_generate_plots(mocker):
    mock_create = mocker.patch("ml_comparison.plotting.create_and_save_scatter_plot")
    generate_plots(datasets_dict)
    assert mock_create.call_count == len(datasets_dict) * 2
