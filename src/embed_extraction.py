# embedding_extraction.py
import os
import torch
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from tqdm import tqdm
from PIL import Image

# Change these model names as needed. (Ensure your GPU has sufficient memory.)
TEXT_MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # or "mistralai/Mistral-7B"
VISION_MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_text_embeddings(input_file, output_file, max_samples=200):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"[INFO] Loading text model and tokenizer for {input_file} ...")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = AutoModel.from_pretrained(TEXT_MODEL_NAME, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to(DEVICE)
    model.eval()

    embeddings = []
    texts = []
    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            texts.append(line.strip())

    print(f"[INFO] Extracting embeddings for {len(texts)} texts from {input_file} ...")
    with torch.no_grad():
        for text in tqdm(texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1].squeeze(0)
            embedding = last_hidden.mean(dim=0).cpu()
            embeddings.append(embedding)
    embeddings = torch.stack(embeddings)
    torch.save({"texts": texts, "embeddings": embeddings}, output_file)
    print(f"[INFO] Text embeddings saved to {output_file}")

def extract_image_embeddings(image_dir, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"[INFO] Loading CLIP model and processor for images in {image_dir} ...")
    processor = CLIPProcessor.from_pretrained(VISION_MODEL_NAME)
    model = CLIPModel.from_pretrained(VISION_MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]
    embeddings = []
    image_paths = []
    print(f"[INFO] Extracting embeddings for {len(image_files)} images from {image_dir} ...")
    with torch.no_grad():
        for img_path in tqdm(image_files):
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model.get_image_features(**inputs)
            embedding = outputs.squeeze(0).cpu()
            embeddings.append(embedding)
            image_paths.append(img_path)
    embeddings = torch.stack(embeddings)
    torch.save({"image_paths": image_paths, "embeddings": embeddings}, output_file)
    print(f"[INFO] Image embeddings saved to {output_file}")

if __name__ == "__main__":
    # Text embeddings: Baseline and Drift
    extract_text_embeddings(input_file="data/text/baseline_text.txt", output_file="data/embeddings/text_embeddings_baseline.pt")
    extract_text_embeddings(input_file="data/text/drift_text.txt", output_file="data/embeddings/text_embeddings_drift.pt")
    
    # Image embeddings: Baseline and Drift
    extract_image_embeddings(image_dir="data/images/baseline", output_file="data/embeddings/image_embeddings_baseline.pt")
    extract_image_embeddings(image_dir="data/images/drift", output_file="data/embeddings/image_embeddings_drift.pt")
