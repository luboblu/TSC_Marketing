from dotenv import load_dotenv
import os
load_dotenv()

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from rag_response import generate_reply

# 環境變數讀值
LINE_TOKEN  = os.getenv("LINE_CHANNEL_TOKEN")
LINE_SECRET = os.getenv("LINE_CHANNEL_SECRET")

app         = Flask(__name__)
line_api    = LineBotApi(LINE_TOKEN)
handler     = WebhookHandler(LINE_SECRET)

@app.route("/callback", methods=["POST"])
def callback():
    sig  = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, sig)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_msg(event):
    txt   = event.message.text
    reply = generate_reply(txt)
    line_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )

if __name__ == "__main__":
    app.run(port=5000)
