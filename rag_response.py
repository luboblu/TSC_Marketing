from dotenv import load_dotenv
import os
load_dotenv()

import openai, faiss
from sentence_transformers import SentenceTransformer
from chat_index import add_chat_to_index

# env 讀值
openai.api_key   = os.getenv("OPENAI_API_KEY")
MODEL_NAME       = os.getenv("EMBEDDING_MODEL_NAME")
EMBED_DIM        = int(os.getenv("EMBED_DIM"))
PRODUCT_INDEX    = os.getenv("PRODUCT_INDEX_PATH")
PRODUCT_ITEMS_TXT= os.getenv("PRODUCT_ITEMS_PATH")

# 載入模型與資料
_model        = SentenceTransformer(MODEL_NAME)
_product_idx  = faiss.read_index(PRODUCT_INDEX)
_items        = open(PRODUCT_ITEMS_TXT, encoding='utf-8').read().splitlines()

def retrieve_top_k(query: str, k: int=3):
    emb = _model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    _, I = _product_idx.search(emb, k)
    return [_items[i] for i in I[0]]

def generate_reply(user_query: str) -> str:
    add_chat_to_index(user_query)
    top = retrieve_top_k(user_query)
    prompt = (
      f"使用者需求：\n{user_query}\n\n"
      "請依據以下商品，用自然、親切的語氣推薦：\n"
      + "\n".join(f"- {p}" for p in top)
    )
    resp = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role":"system","content":"你是行銷專家"},
        {"role":"user","content":prompt}
      ]
    )
    return resp.choices[0].message.content
