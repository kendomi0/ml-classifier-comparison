import matplotlib.pyplot as plt

def create_scatter_plot(X, y, current_dataset):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='plasma', edgecolors='k')
    plt.title(f"{current_dataset.title()} Dataset")
    plt.show()

def trigger_plot_creation(X, y, current_dataset):
    input_msg = "Would you like to see a scatter plot of the data? (Y or N): "
    display_plot = input(input_msg).capitalize()
    while display_plot not in ["Y", "N"]:
        print("Invalid input.")
        display_plot = input(input_msg).capitalize()
    if display_plot == "Y":
        create_scatter_plot(X, y, current_dataset)