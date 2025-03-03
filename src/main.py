# main.py
import os
import subprocess

def run_script(script_name):
    print(f"\n=== Running {script_name} ===")
    subprocess.run(["python", script_name], check=True)

if __name__ == "__main__":
    # 1. Download datasets
    # run_script("data_download.py")
    
    # 2. Extract embeddings for text and images
    run_script("embedding_extraction.py")
    
    # 3. Compute drift metrics
    run_script("drift_detection.py")
    
    # 4. Visualize embeddings
    run_script("visualization.py")
    
    print("\nPipeline complete!")
