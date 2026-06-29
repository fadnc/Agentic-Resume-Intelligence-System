import os
from dotenv import load_dotenv
load_dotenv()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

USE_SAGEMAKER      = os.getenv("USE_SAGEMAKER", "False").lower() == "true"
SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
USE_GROQ           = bool(GROQ_API_KEY) and not USE_SAGEMAKER

FAISS_PATH   = "embeddings_store/faiss_index"
META_PATH    = "embeddings_store/meta.npy"
KB_STORE_DIR = "embeddings_store/knowledge_base"
KB_SOURCE_DIR = "knowledge_base"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT   = "document-intelligence-v7"