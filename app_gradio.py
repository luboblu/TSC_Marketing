import os
import numpy as np
import faiss
from openai import OpenAI
from io import BytesIO
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import torch
import warnings
from dotenv import load_dotenv
import gradio as gr

# 載入環境變數
load_dotenv()
# 初始化 OpenAI 客戶端
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# 指定運算裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 步驟1：讀取並切分商品資料庫
def load_and_partition_text(file_path, chunk_size=300, chunk_overlap=50):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    sections = content.split('---')
    partitioned = {}
    for sec in sections:
        if sec.strip():
            lines = sec.strip().splitlines()
            header = lines[0].replace('名稱:', '').strip()
            body = '\n'.join(lines[1:]).strip()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n", ".", "。"]
            )
            docs = [Document(page_content=body)]
            splits = splitter.split_documents(docs)
            segments = [f"{header}: {d.page_content}" for d in splits]
            partitioned[header] = segments
    return partitioned

# 步驟2：初始化 FAISS 索引與文本嵌入
def initialize_rag(file_path):
    partitions = load_and_partition_text(file_path)
    model = SentenceTransformer("intfloat/multilingual-e5-base", device=device)
    indexes, segments_map = {}, {}
    for header, segs in partitions.items():
        embeddings = model.encode(segs, batch_size=8, show_progress_bar=False)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))
        indexes[header] = index
        segments_map[header] = segs
    return model, indexes, segments_map

# 載入並初始化資料庫
model, indexes, segments_map = initialize_rag('product_items.txt')

# 查詢 RAG
def query_rag(query, top_k=3, threshold=1.5):
    q_emb = model.encode([query], show_progress_bar=False)
    results = []
    for hdr, index in indexes.items():
        D, I = index.search(np.array(q_emb), k=top_k)
        for dist, idx in zip(D[0], I[0]):
            if idx != -1 and dist < threshold:
                results.append(segments_map[hdr][idx])
    return results if results else ["很抱歉，找不到相關產品資訊。"]

def generate_answer(query, contexts):
    context = "".join(contexts)
    # 將 system_prompt 變成一個單一的長字串
    system_prompt = """你是一個運動商品行銷助手，負責提供全越運動營養、配件與周邊商品的推薦，並幫助品牌完成行銷轉換。
1. 如果使用者的問題含有「系統在做什麼」、「這個系統」等相關字眼，
   負責提供全越運動營養、配件與周邊商品的推薦，讓使用者能夠體驗最適配的產品。
2. 如果使用者的問題是詢問「推薦產品」或具體的產品需求，
   請依據上下文提供最多五項重點推薦，使用清晰條列，格式如下：
     序號. 名稱（價格）：特色 + 建議購買量  
   如接近 token 限制，最後加註「以上說明完畢，如需更多資訊請告知。」"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"使用者問：{query}\n相關上下文：{context}"}
    ]
    resp = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0.7,
        max_tokens=600
    )
    content = resp.choices[0].message.content.strip()

    # 若回覆未以標點結尾，自動續接
    if not content.endswith(("。", "！", "？", ".")):
        follow_resp = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "請接續未完成部分，繼續完成回覆。"}
            ],
            temperature=0.7,
            max_tokens=200
        )
        content += follow_resp.choices[0].message.content.strip()

    return content



# 文字聊天
def chat_text(user_input):
    segs = query_rag(user_input)
    return generate_answer(user_input, segs)

# 語音轉文字並聊天
def chat_audio(audio_file):
    if not audio_file:
        return ""
    try:
        with open(audio_file, 'rb') as af:
            resp = openai_client.audio.transcriptions.create(model="whisper-1", file=af)
        transcription = resp.text
    except Exception as e:
        return f"語音轉錄失敗：{e}"
    return chat_text(transcription)

# Gradio 介面設置
with gr.Blocks() as demo:
    gr.Markdown("### 全越AI動管家")
    with gr.Row():
        txt = gr.Textbox(label="輸入您的問題（文字）")
        mic = gr.Microphone(label="語音輸入（講完即辨識）", type="filepath")
    out = gr.Textbox(label="回覆")

    txt.submit(chat_text, txt, out)
    mic.change(chat_audio, mic, out)

    gr.Markdown("---\nPowered by RAG + GPT-4")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
