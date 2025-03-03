# drift_detection.py
import torch
import numpy as np
from scipy.stats import wasserstein_distance
from bert_score import score as bertscore
from tqdm import tqdm

def compute_bertscore_drift(baseline_texts, drift_texts):
    """
    Computes average BERTScore F1 between baseline and drift texts.
    Lower F1 indicates greater drift.
    """
    P, R, F1 = bertscore(drift_texts, baseline_texts, lang="en", verbose=True)
    avg_f1 = F1.mean().item()
    print(f"[INFO] Average BERTScore F1 (Text): {avg_f1:.4f}")
    return avg_f1

def compute_wasserstein_drift(baseline_embeddings, drift_embeddings):
    """
    Computes the average Wasserstein distance over dimensions.
    Works for both text and image embeddings.
    """
    baseline_np = baseline_embeddings.numpy()
    drift_np = drift_embeddings.numpy()
    distances = [wasserstein_distance(baseline_np[:, i], drift_np[:, i])
                 for i in range(baseline_np.shape[1])]
    avg_distance = np.mean(distances)
    print(f"[INFO] Average Wasserstein Distance: {avg_distance:.4f}")
    return avg_distance

def compute_alignment_loss(baseline_embeddings, drift_embeddings):
    """
    Computes alignment loss as the Frobenius norm between mean-centered embeddings.
    Works for both text and image embeddings.
    """
    baseline_centered = baseline_embeddings - baseline_embeddings.mean(dim=0)
    drift_centered = drift_embeddings - drift_embeddings.mean(dim=0)
    loss = torch.norm(baseline_centered - drift_centered, p='fro').item()
    print(f"[INFO] Alignment Loss (Frobenius norm): {loss:.4f}")
    return loss

if __name__ == "__main__":
    #### TEXT DRIFT ANALYSIS ####
    baseline_text_data = torch.load("data/embeddings/text_embeddings_baseline.pt")
    drift_text_data = torch.load("data/embeddings/text_embeddings_drift.pt")
    baseline_texts = baseline_text_data["texts"]
    drift_texts = drift_text_data["texts"]
    baseline_text_embeddings = baseline_text_data["embeddings"]
    drift_text_embeddings = drift_text_data["embeddings"]

    print("\n=== TEXT DRIFT METRICS ===")
    compute_bertscore_drift(baseline_texts, drift_texts)
    compute_wasserstein_drift(baseline_text_embeddings, drift_text_embeddings)
    compute_alignment_loss(baseline_text_embeddings, drift_text_embeddings)

    #### IMAGE DRIFT ANALYSIS ####
    baseline_image_data = torch.load("data/embeddings/image_embeddings_baseline.pt")
    drift_image_data = torch.load("data/embeddings/image_embeddings_drift.pt")
    baseline_image_embeddings = baseline_image_data["embeddings"]
    drift_image_embeddings = drift_image_data["embeddings"]

    print("\n=== IMAGE DRIFT METRICS ===")
    compute_wasserstein_drift(baseline_image_embeddings, drift_image_embeddings)
    compute_alignment_loss(baseline_image_embeddings, drift_image_embeddings)

    print("\n[INFO] Real drift analysis completed for both text and image embeddings.")
