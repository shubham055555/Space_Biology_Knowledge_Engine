# src/02_generate_embeddings.py
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
data_file = r"C:\Users\Krishna\Downloads\OSD-101_metadata_OSD-101-ISA\OSD-101_clean.csv"
artifacts_path = os.path.join("artifacts")
os.makedirs(artifacts_path, exist_ok=True)
emb_file = os.path.join(artifacts_path, "embeddings.npy")

# Load dataset
df = pd.read_csv(data_file)
df["Summary"] = df["Summary"].fillna("")

# Load embedding model
print("🔄 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
print("⚡ Generating embeddings...")
embeddings = model.encode(df["Summary"].tolist(), convert_to_numpy=True, show_progress_bar=True)

# Save
np.save(emb_file, embeddings.astype("float32"))
print(f"✅ Embeddings saved at {emb_file}")