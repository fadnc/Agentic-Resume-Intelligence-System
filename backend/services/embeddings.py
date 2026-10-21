from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from collections import OrderedDict
import hashlib
import threading
from backend.config import EMBED_MODEL

_model = None
_model_lock = threading.Lock()
_vector_cache = OrderedDict()
_vector_cache_lock = threading.Lock()
_VECTOR_CACHE_SIZE = 1024

def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _model = SentenceTransformer(EMBED_MODEL, device=device)
    return _model

def embed_texts(texts):
    texts = [str(text) for text in texts]
    if not texts:
        return np.empty((0, 0), dtype="float32")

    keys = [hashlib.sha256((EMBED_MODEL + "\0" + text).encode("utf-8")).hexdigest() for text in texts]
    vectors_by_key = {}
    missing = {}
    with _vector_cache_lock:
        for key, text in zip(keys, texts):
            cached = _vector_cache.get(key)
            if cached is not None:
                _vector_cache.move_to_end(key)
                vectors_by_key[key] = cached.copy()
            else:
                missing.setdefault(key, text)

    if missing:
        encoded = get_model().encode(
            list(missing.values()),
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,
            normalize_embeddings=True,
        ).astype("float32")
        with _vector_cache_lock:
            for key, vector in zip(missing, encoded):
                existing = _vector_cache.get(key)
                if existing is None:
                    _vector_cache[key] = vector.copy()
                    existing = _vector_cache[key]
                _vector_cache.move_to_end(key)
                vectors_by_key[key] = existing.copy()
            while len(_vector_cache) > _VECTOR_CACHE_SIZE:
                _vector_cache.popitem(last=False)

    return np.stack([vectors_by_key[key] for key in keys]).astype("float32", copy=False)
