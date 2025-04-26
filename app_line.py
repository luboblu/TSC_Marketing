import os
import numpy as np
import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import torch
import warnings
from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, AudioMessage, TextSendMessage

# 載入環境變數
load_dotenv()

# 初始化 OpenAI 客戶端
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# LINE Bot 初始化
line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

# 指定運算裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 拆分長文字為多段，避免訊息過長
def split_text(text, max_len=1000):
    sentences = text.replace("。", "。").split("\n")
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_len:
            chunks.append(current)
            current = sent
        else:
            current += ("" if not current else "") + sent
    if current:
        chunks.append(current)
    return chunks

# 讀取並切分商品資料庫
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

# 初始化 RAG
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

# 生成回答並附結尾提示
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

# 回覆處理

def reply_text(token, text):
    if len(text) <= 1000:
        line_bot_api.reply_message(token, TextSendMessage(text=text))
    else:
        parts = split_text(text)
        messages = [TextSendMessage(text=p) for p in parts]
        line_bot_api.reply_message(token, messages)

# 處理文字事件
def handle_text(event):
    segs = query_rag(event.message.text)
    reply = generate_answer(event.message.text, segs)
    reply_text(event.reply_token, reply)

# 處理語音事件
def handle_audio(event):
    msg = line_bot_api.get_message_content(event.message.id)
    tmp_file = f"tmp_{event.message.id}.m4a"
    # 下載並存檔
    with open(tmp_file, 'wb') as f:
        for chunk in msg.iter_content():
            f.write(chunk)
    # 使用 with 確保檔案關閉後再刪除
    with open(tmp_file, 'rb') as audio_f:
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1", file=audio_f
        ).text
    # 讀取完畢後關閉，再刪除檔案
    os.remove(tmp_file)
    # 產生並回覆
    reply = generate_answer(transcription, query_rag(transcription))
    reply_text(event.reply_token, reply)

# Flask 應用與路由
app = Flask(__name__)
@app.route('/callback', methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# 綁定事件
@handler.add(MessageEvent, message=TextMessage)
def on_text(event): handle_text(event)

@handler.add(MessageEvent, message=AudioMessage)
def on_audio(event): handle_audio(event)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    print(f"啟動 LINE Bot，監聽 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port)
