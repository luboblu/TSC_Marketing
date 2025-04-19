from dotenv import load_dotenv
import os
load_dotenv()

from sentence_transformers import SentenceTransformer
import faiss

MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
EMBED_DIM  = int(os.getenv("EMBED_DIM"))

_model     = SentenceTransformer(MODEL_NAME)
_chat_idx  = faiss.IndexFlatIP(EMBED_DIM)

def add_chat_to_index(text: str):
    emb = _model.encode([text], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    _chat_idx.add(emb)

def get_chat_index():
    return _chat_idx
