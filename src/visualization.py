# visualization.py
import os
import torch
import matplotlib.pyplot as plt
import umap.umap_ as umap

def visualize_embeddings(embeddings, title="Embedding Visualization", save_path=None):
    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(embeddings.numpy())
    plt.figure(figsize=(8,6))
    plt.scatter(embedding_2d[:,0], embedding_2d[:,1], s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Plot saved to {save_path}")
    plt.show()

def process_and_plot_text_embeddings():
    baseline_data = torch.load("data/embeddings/text_embeddings_baseline.pt")
    drift_data = torch.load("data/embeddings/text_embeddings_drift.pt")
    baseline_embeddings = baseline_data["embeddings"]
    drift_embeddings = drift_data["embeddings"]

    visualize_embeddings(baseline_embeddings, title="Baseline Text Embeddings", save_path="plots/baseline_text_embeddings.png")
    visualize_embeddings(drift_embeddings, title="Drift Text Embeddings", save_path="plots/drift_text_embeddings.png")

def process_and_plot_image_embeddings():
    baseline_data = torch.load("data/embeddings/image_embeddings_baseline.pt")
    drift_data = torch.load("data/embeddings/image_embeddings_drift.pt")
    baseline_embeddings = baseline_data["embeddings"]
    drift_embeddings = drift_data["embeddings"]

    visualize_embeddings(baseline_embeddings, title="Baseline Image Embeddings", save_path="plots/baseline_image_embeddings.png")
    visualize_embeddings(drift_embeddings, title="Drift Image Embeddings", save_path="plots/drift_image_embeddings.png")

if __name__ == "__main__":
    print("\n=== Visualizing Text Embeddings ===")
    process_and_plot_text_embeddings()

    print("\n=== Visualizing Image Embeddings ===")
    process_and_plot_image_embeddings()

    print("\n[INFO] Visualization complete. Check the 'plots/' directory for output images.")
