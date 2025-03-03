# data_download.py
import os
from datasets import load_dataset
from torchvision import datasets, transforms
from PIL import Image

def download_baseline_text_dataset(save_dir="data/text"):
    os.makedirs(save_dir, exist_ok=True)
    print("[INFO] Loading baseline text dataset (wikitext-2-raw-v1, 500 samples)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:500]")
    texts = dataset["text"]
    with open(os.path.join(save_dir, "baseline_text.txt"), "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text.replace("\n", " ") + "\n")
    print(f"[INFO] Saved {len(texts)} baseline text samples to {save_dir}/baseline_text.txt")

def download_drift_text_dataset(save_dir="data/text"):
    os.makedirs(save_dir, exist_ok=True)
    print("[INFO] Loading drift text dataset (AG News, 500 samples)...")
    dataset = load_dataset("ag_news", split="train[:500]")
    texts = dataset["text"]
    with open(os.path.join(save_dir, "drift_text.txt"), "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text.replace("\n", " ") + "\n")
    print(f"[INFO] Saved {len(texts)} drift text samples to {save_dir}/drift_text.txt")

def download_baseline_image_dataset(save_dir="data/images/baseline"):
    os.makedirs(save_dir, exist_ok=True)
    print("[INFO] Downloading baseline image dataset (CIFAR10, 100 images)...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    cifar = datasets.CIFAR10(root="data/cifar", train=True, download=True, transform=transform)
    for i in range(100):
        img, label = cifar[i]
        img_path = os.path.join(save_dir, f"img_{i}.png")
        img_pil = transforms.ToPILImage()(img)
        img_pil.save(img_path)
    print(f"[INFO] Saved 100 baseline images to {save_dir}")

def download_drift_image_dataset(save_dir="data/images/drift"):
    os.makedirs(save_dir, exist_ok=True)
    print("[INFO] Downloading drift image dataset (CIFAR10 with color jitter, 100 images)...")
    # Apply a color jitter transformation to simulate a real-world shift (e.g., different lighting or camera conditions)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.ToTensor()
    ])
    cifar = datasets.CIFAR10(root="data/cifar", train=True, download=True, transform=transform)
    for i in range(100):
        img, label = cifar[i]
        img_path = os.path.join(save_dir, f"img_{i}.png")
        img_pil = transforms.ToPILImage()(img)
        img_pil.save(img_path)
    print(f"[INFO] Saved 100 drift images to {save_dir}")

if __name__ == "__main__":
    download_baseline_text_dataset()
    download_drift_text_dataset()
    download_baseline_image_dataset()
    download_drift_image_dataset()
