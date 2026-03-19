import sys
sys.path.insert(0, ".")

from backend.services.corpus_builder import load_all_corpora

load_all_corpora()
print("Knowledge base indexed successfully.")