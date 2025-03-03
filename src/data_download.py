# data_download.py
import os
from datasets import load_dataset
from torchvision import datasets, transforms
from PIL import Image

def download_text_dataset(save_dir="data/text"):
    os.makedirs(save_dir, exist_ok=True)
    print("[INFO] Starting to load OpenWebText (500 samples)...")
    dataset = load_dataset("stas/openwebtext-10k", split="train[:500]", trust_remote_code=True)
    print("[INFO] Successfully loaded dataset!")

    texts = dataset["text"]
    with open(os.path.join(save_dir, "openwebtext_samples.txt"), "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text.replace("\n", " ") + "\n")
    print(f"[INFO] Saved {len(texts)} text samples to {save_dir}/openwebtext_samples.txt")


def download_image_dataset(save_dir="data/images"):
    os.makedirs(save_dir, exist_ok=True)
    # For demonstration, we use CIFAR10. Replace this with LAION image loading if available.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    cifar = datasets.CIFAR10(root="data/cifar", train=True, download=False, transform=transform)

    # Save first 100 images
    for i in range(100):
        img, label = cifar[i]
        img_path = os.path.join(save_dir, f"img_{i}.png")
        # Convert tensor to PIL image and save
        img_pil = transforms.ToPILImage()(img)
        img_pil.save(img_path)
    print(f"Saved 100 images to {save_dir}")

if __name__ == "__main__":
    # download_text_dataset()
    download_image_dataset()
