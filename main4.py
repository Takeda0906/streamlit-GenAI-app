import os
import streamlit as st
import tiktoken
from langchain.chat_models.openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langsmith import Client

# ==== LangSmith ログ設定 ====
def setup_langsmith():
    api_key = os.getenv("LANGCHAIN_API_KEY")
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"

    st.sidebar.markdown("## 🧠 LangSmith ログ設定")
    if api_key and tracing_enabled:
        st.sidebar.success("✅ LangSmith ログ送信が有効です。")
        client = Client(api_key=api_key)
        st.session_state["langsmith_client"] = client
    else:
        st.sidebar.warning("⚠ LangSmith ログ送信が無効です。")
        st.sidebar.markdown("""
        - `LANGCHAIN_API_KEY` が設定されているか確認してください  
        - `LANGCHAIN_TRACING_V2=true` を環境変数に設定してください
        """)

# ==== モデル別価格設定 ====
MODEL_PRICES = {
    "input": {
        "gpt-3.5-turbo": 0.5 / 1_000_000,
        "gpt-4o": 5 / 1_000_000,
        "gpt-5": 10 / 1_000_000,
        "gpt-5-mini": 2 / 1_000_000,
        "claude-3-haiku-20240307": 0.25 / 1_000_000,
        "gemini-2.5-pro": 3.5 / 1_000_000,
        "gemini-2.5-flash": 0.075 / 1_000_000,
    },
    "output": {
        "gpt-3.5-turbo": 1.5 / 1_000_000,
        "gpt-4o": 15 / 1_000_000,
        "gpt-5": 30 / 1_000_000,
        "gpt-5-mini": 6 / 1_000_000,
        "claude-3-haiku-20240307": 1.25 / 1_000_000,
        "gemini-2.5-pro": 10.5 / 1_000_000,
        "gemini-2.5-flash": 0.30 / 1_000_000,
    },
}

# ==== 初期化 ====
def init_page():
    st.set_page_config(page_title="AI Chat App", page_icon="🤖")
    st.header("AI Chat App 🤖")
    st.sidebar.title("設定")
    setup_langsmith()

def init_messages():
    if "message_history" not in st.session_state:
        st.session_state.message_history = [("system", "You are a helpful assistant.")]
    if st.sidebar.button("💬 会話をリセット"):
        st.session_state.message_history = [("system", "You are a helpful assistant.")]

# ==== LLM初期化 ====
def create_llm(model_choice, temperature):
    model_map = {
        "GPT-3.5": "gpt-3.5-turbo",
        "GPT-4": "gpt-4o",
        "GPT-5": "gpt-5",
        "GPT-5 Mini": "gpt-5-mini",
        "Claude 3 Haiku": "claude-3-haiku-20240307",
        "Gemini 2.5 Pro": "gemini-2.5-pro",
        "Gemini 2.5 Flash": "gemini-2.5-flash",
    }

    model_name = model_map[model_choice]
    st.session_state.model_name = model_name

    try:
        if "gpt" in model_name:
            return ChatOpenAI(model_name=model_name, temperature=temperature)
        elif "claude" in model_name:
            return ChatAnthropic(model=model_name, temperature=temperature)
        elif "gemini" in model_name:
            return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    except Exception as e:
        st.error(f"モデル初期化失敗: {e}")
        return None

# ==== モデル選択 ====
def select_model():
    model_options = [
        "GPT-3.5",
        "GPT-4",
        "GPT-5",
        "GPT-5 Mini",
        "Claude 3 Haiku",
        "Gemini 2.5 Pro",
        "Gemini 2.5 Flash",
    ]

    default_model = st.session_state.get("model_choice", "GPT-3.5")
    default_temp = st.session_state.get("temperature", 0.7)

    model_choice = st.sidebar.radio(
        "使用するモデルを選択:",
        model_options,
        index=model_options.index(default_model),
    )

    if "model_choice" not in st.session_state or st.session_state.model_choice != model_choice:
        st.session_state.model_choice = model_choice
        st.rerun()

    if model_choice in ["GPT-5", "GPT-5 Mini"]:
        st.sidebar.info("⚠ GPT-5 系モデルは固定温度 1 のみ使用可能です。")
        temperature = 1.0
    elif "Claude" in model_choice:
        temperature = st.sidebar.slider(
            "温度 (創造性):", 0.0, 1.0, default_temp, 0.01, key="temp_claude"
        )
    else:
        temperature = st.sidebar.slider(
            "温度 (創造性):", 0.0, 2.0, default_temp, 0.01, key="temp_other"
        )

    st.session_state.temperature = float(temperature)
    st.session_state.llm = create_llm(model_choice, temperature)

# ==== トークンカウント ====
def get_token_count(text, model_name):
    if "claude" in model_name or "gemini" in model_name:
        return len(text) // 2
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
        ret

