# visualization.py
import os
import torch
import matplotlib.pyplot as plt
import umap.umap_ as umap
from drift_detection import simulate_drift

def visualize_embeddings(embeddings, title="Embedding Visualization", save_path=None):
    """
    Embedding visualization using UMAP.

    Args:
        embeddings (torch.Tensor): The embedding matrix [N, D].
        title (str): Title of the plot.
        save_path (str): Path to save the plot (e.g., "../plots/embedding_plot.png").
    """
    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(embeddings.numpy())

    plt.figure(figsize=(8, 6))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()

def process_and_plot_text_embeddings():
    """
    Loads text embeddings, simulates drift, and plots baseline + drifted embeddings.
    """
    data = torch.load("data/embeddings/text_embeddings.pt")
    baseline_embeddings = data["embeddings"]

    # Simulate drift for visualization
    drifted_embeddings = simulate_drift(baseline_embeddings, drift_strength=0.1)

    visualize_embeddings(
        baseline_embeddings,
        title="Baseline Text Embeddings",
        save_path="plots/baseline_text_embeddings.png"
    )

    visualize_embeddings(
        drifted_embeddings,
        title="Drifted Text Embeddings",
        save_path="plots/drifted_text_embeddings.png"
    )

def process_and_plot_image_embeddings():
    """
    Loads image embeddings, simulates drift, and plots baseline + drifted embeddings.
    """
    data = torch.load("data/embeddings/image_embeddings.pt")
    baseline_embeddings = data["embeddings"]

    # Simulate drift for visualization
    drifted_embeddings = simulate_drift(baseline_embeddings, drift_strength=0.1)

    visualize_embeddings(
        baseline_embeddings,
        title="Baseline Image Embeddings",
        save_path="plots/baseline_image_embeddings.png"
    )

    visualize_embeddings(
        drifted_embeddings,
        title="Drifted Image Embeddings",
        save_path="plots/drifted_image_embeddings.png"
    )

if __name__ == "__main__":
    print("\n=== Visualizing Text Embeddings ===")
    process_and_plot_text_embeddings()

    print("\n=== Visualizing Image Embeddings ===")
    process_and_plot_image_embeddings()

    print("\nVisualization complete. Check 'plots/' directory for images.")
