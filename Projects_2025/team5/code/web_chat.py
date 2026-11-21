import requests
import gradio as gr
import time
import os

# ======================
# 模型配置
# ======================
MODEL_CONFIG = {
    "llama": {
        "url": "http://127.0.0.1:9100/v1/chat/completions",
        "model_name": "llama4-dolphin-8b",
        "temperature": 0.8,
    },
    "qwen": {
        "url": "http://127.0.0.1:8090/v1/chat/completions",
        "model_name": "Qwen3-1.7B",
        "temperature": 0.7,
    }
}

MAX_HISTORY = 8  # 每个模型保留多少条消息
chat_history = {key: [] for key in MODEL_CONFIG.keys()}

# ======================
# 提取模型返回
# ======================
def extract_reply(data):
    # 1️⃣ 优先取 choices -> message -> content
    if "choices" in data and len(data["choices"]) > 0:
        choice = data["choices"][0]
        if "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]
        if "text" in choice:
            return choice["text"]
    # 2️⃣ 一些 Qwen 接口旧版本可能返回 response/output/output_text/answer
    for key in ["response","output","output_text","text","answer"]:
        if key in data:
            return data[key]
    # 3️⃣ 如果仅返回 ok: True -> 无内容
    if "ok" in data and data["ok"] is True:
        return "✅ Qwen 接口返回 ok，但没有内容，请确认模型已加载并生成回答"
    # 4️⃣ error
    if "error" in data:
        return f"❌ API Error: {data['error']}"
    return f"❌ Unknown response:\n{data}"

# ======================
# 流式聊天函数
# ======================
def chat_stream(message, model_key):
    if not message.strip():
        yield chat_history[model_key], ""
        return

    history = chat_history[model_key][-MAX_HISTORY:]
    messages = []
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": message})

    cfg = MODEL_CONFIG[model_key]
    payload = {"model": cfg["model_name"], "messages": messages, "temperature": cfg["temperature"]}

    try:
        res = requests.post(cfg["url"], json=payload, timeout=120)
        res.raise_for_status()
        data = res.json()
        reply_full = extract_reply(data)
    except Exception as e:
        reply_full = f"❌ Request failed: {e}"

    # 添加用户消息到历史
    chat_history[model_key].append((message, ""))  # 初始空回复
    yield chat_history[model_key], ""  # 立即清空输入框

    # 流式输出
    chunk_size = 30
    reply_display = ""
    for i in range(0, len(reply_full), chunk_size):
        reply_display += reply_full[i:i+chunk_size]
        chat_history[model_key][-1] = (message, reply_display)
        time.sleep(0.02)
        yield chat_history[model_key], ""  # 输入框保持清空

    # 保存最终回复
    chat_history[model_key][-1] = (message, reply_full)

# ======================
# Gradio 页面
# ======================
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

with gr.Blocks(title="🤖 快来和大模型对话吧！") as demo:
    # manifest + theme + service worker
    gr.HTML(f"""
    <link rel="manifest" href="/static/manifest.json">
    <meta name="theme-color" content="#4F46E5">
    <script>
    if ('serviceWorker' in navigator) {{
        navigator.serviceWorker.register('/static/service-worker.js')
        .then(() => console.log('Service Worker registered'));
    }}
    </script>
    """)

    gr.Markdown("## 🤖 快来和大模型对话吧！")
    model_selector = gr.Dropdown(list(MODEL_CONFIG.keys()), value="llama", label="选择模型")
    chatbot = gr.Chatbot(height=420)
    msg = gr.Textbox(placeholder="输入你的问题，回车发送")
    clear = gr.Button("清空当前模型对话")

    # 提交消息：流式输出 + 输入框自动清空
    msg.submit(chat_stream, inputs=[msg, model_selector], outputs=[chatbot, msg])

    # 清空历史
    def clear_history(model_key):
        chat_history[model_key] = []
        return [], ""
    clear.click(clear_history, inputs=[model_selector], outputs=[chatbot, msg])

# 固定端口 7861
demo.launch(server_name="0.0.0.0", server_port=7861, show_error=True)
