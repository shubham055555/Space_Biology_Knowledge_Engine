# src/03_build_index_and_save.py
import os
import numpy as np

emb_file = "artifacts/embeddings.npy"
faiss_out = "artifacts/faiss.index"

if not os.path.exists(emb_file):
    raise FileNotFoundError("❌ Embeddings not found. Run step 02 first.")

# Load embeddings
emb = np.load(emb_file).astype("float32")

try:
    import faiss
    # Normalize vectors (for cosine similarity)
    faiss.normalize_L2(emb)
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    faiss.write_index(index, faiss_out)
    print("✅ FAISS index saved to:", faiss_out)
except Exception as e:
    print("⚠️ FAISS not available. Using numpy fallback. Error:", e)
