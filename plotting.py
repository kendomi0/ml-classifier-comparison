import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

def create_and_save_plot(X, y, current_dataset):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='plasma', edgecolors='k')
    plt.title(f"{current_dataset.title()} Dataset")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf