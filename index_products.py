from dotenv import load_dotenv
import os
load_dotenv()

from sentence_transformers import SentenceTransformer
import faiss, numpy as np

# 直接從 env 讀
MODEL_NAME       = os.getenv("EMBEDDING_MODEL_NAME")
EMBED_DIM        = int(os.getenv("EMBED_DIM"))
PRODUCT_INDEX    = os.getenv("PRODUCT_INDEX_PATH")
PRODUCT_EMBS     = os.getenv("PRODUCT_EMBS_PATH")
PRODUCT_ITEMS_TXT= os.getenv("PRODUCT_ITEMS_PATH")

def load_product_txt(path=PRODUCT_ITEMS_TXT):
    with open(path, encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]

def build_product_index():
    model = SentenceTransformer(MODEL_NAME)
    items = load_product_txt()
    embs  = model.encode(items, convert_to_numpy=True)
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embs)
    faiss.write_index(index, PRODUCT_INDEX)
    np.save(PRODUCT_EMBS, embs)
    return index, items

if __name__ == "__main__":
    build_product_index()
