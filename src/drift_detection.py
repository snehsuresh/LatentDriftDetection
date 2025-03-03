# drift_detection.py
import torch
import numpy as np
from scipy.stats import wasserstein_distance
from bert_score import score as bertscore
from tqdm import tqdm

### TEXT DRIFT FUNCTIONS ###

def compute_bertscore_drift(baseline_texts, new_texts):
    """
    Computes average BERTScore F1 between baseline and new texts.
    Lower F1 suggests greater drift.
    """
    P, R, F1 = bertscore(new_texts, baseline_texts, lang="en", verbose=True)
    avg_f1 = F1.mean().item()
    print(f"Average BERTScore F1 (Text): {avg_f1:.4f}")
    return avg_f1

### SHARED DRIFT FUNCTIONS (APPLIES TO BOTH TEXT AND IMAGES) ###

def compute_wasserstein_drift(baseline_embeddings, new_embeddings):
    """
    Computes average Wasserstein distance over dimensions.
    Works for both text and image embeddings.
    """
    baseline_np = baseline_embeddings.numpy()
    new_np = new_embeddings.numpy()

    distances = [
        wasserstein_distance(baseline_np[:, i], new_np[:, i])
        for i in range(baseline_np.shape[1])
    ]

    avg_distance = np.mean(distances)
    print(f"Average Wasserstein Distance: {avg_distance:.4f}")
    return avg_distance

def compute_alignment_loss(baseline_embeddings, new_embeddings):
    """
    Computes alignment loss (Frobenius norm between mean-centered embeddings).
    Works for both text and image embeddings.
    """
    baseline_centered = baseline_embeddings - baseline_embeddings.mean(dim=0)
    new_centered = new_embeddings - new_embeddings.mean(dim=0)
    loss = torch.norm(baseline_centered - new_centered, p='fro').item()
    print(f"Alignment Loss (Frobenius norm): {loss:.4f}")
    return loss

def simulate_drift(embeddings, drift_strength=0.05):
    """
    Simulates drift by adding small random perturbations to embeddings.
    Works for both text and image embeddings.
    """
    noise = drift_strength * torch.randn_like(embeddings)
    return embeddings + noise

### MAIN (EXECUTION ENTRY POINT) ###

if __name__ == "__main__":
    #### TEXT EMBEDDING DRIFT ANALYSIS ####
    text_data = torch.load("data/embeddings/text_embeddings.pt")
    baseline_texts = text_data["texts"]
    baseline_text_embeddings = text_data["embeddings"]

    # Simulate drift for text
    drifted_text_embeddings = simulate_drift(baseline_text_embeddings, drift_strength=0.1)

    # For BERTScore, also create new (drifted) texts - word swaps as fake semantic drift
    new_texts = []
    for text in baseline_texts:
        words = text.split()
        if len(words) > 5:
            words[1], words[-2] = words[-2], words[1]
        new_texts.append(" ".join(words))

    print("\n=== TEXT DRIFT METRICS ===")
    compute_bertscore_drift(baseline_texts, new_texts)
    compute_wasserstein_drift(baseline_text_embeddings, drifted_text_embeddings)
    compute_alignment_loss(baseline_text_embeddings, drifted_text_embeddings)

    #### IMAGE EMBEDDING DRIFT ANALYSIS ####
    image_data = torch.load("data/embeddings/image_embeddings.pt")
    baseline_image_embeddings = image_data["embeddings"]

    # Simulate drift for images
    drifted_image_embeddings = simulate_drift(baseline_image_embeddings, drift_strength=0.1)

    print("\n=== IMAGE DRIFT METRICS ===")
    compute_wasserstein_drift(baseline_image_embeddings, drifted_image_embeddings)
    compute_alignment_loss(baseline_image_embeddings, drifted_image_embeddings)

    print("\nDrift analysis completed for both text and image embeddings.")
