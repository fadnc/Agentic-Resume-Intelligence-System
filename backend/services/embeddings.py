from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from backend.config import EMBED_MODEL

_model = None

def get_model():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(EMBED_MODEL, device=device)
    return _model

def embed_texts(texts):
    model = get_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=64,
        normalize_embeddings=True
    )
    return np.array(vectors).astype("float32")