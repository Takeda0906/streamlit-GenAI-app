import streamlit as st
import tiktoken

# ==== 最新 LangChain 1.x 用 Import ====
from langchain.chat_models.openai import ChatOpenAI
from langchain.chat_models.anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# ==== モデル別価格設定（USD/1M tokens） ====
MODEL_PRICES = {
    "input": {
        "gpt-3.5-turbo": 0.5 / 1_000_000,
        "gpt-4o": 5 / 1_000_000,
        "gpt-5": 10 / 1_000_000,
        "gpt-5-mini": 2 / 1_000_000,
        "claude-3-haiku-20240307": 0.25 / 1_000_000,
        "gemini-2.5-pro": 3.5 / 1_000_000,
        "gemini-2.5-flash": 0.075 / 1_000_000
    },
    "output": {
        "gpt-3.5-turbo": 1.5 / 1_000_000,
        "gpt-4o": 15 / 1_000_000,
        "gpt-5": 30 / 1_000_000,
        "gpt-5-mini": 6 / 1_000_000,
        "claude-3-haiku-20240307": 1.25 / 1_000_000,
        "gemini-2.5-pro": 10.5 / 1_000_000,
        "gemini-2.5-flash": 0.30 / 1_000_000
    }
}

# ==== ページ初期化 ====
def init_page():
    st.set_page_config(page_title="AI Chat App", page_icon="🤖")
    st.header("AI Chat App 🤖")
    st.sidebar.title("設定")

# ==== メッセージ履歴初期化 ====
def init_messages():
    if "message_history" not in st.session_state:
        st.session_state.message_history = [("system", "You are a helpful assistant.")]
    if st.sidebar.button("💬 会話をリセット"):
        st.session_state.message_history = [("system", "You are a helpful assistant.")]

# ==== モデル選択 ====
def select_model():
    model_options = [
        "GPT-3.5", "GPT-4", "GPT-5", "GPT-5 Mini",
        "Claude 3 Haiku", "Gemini 2.5 Pro", "Gemini 2.5 Flash"
    ]

    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "GPT-3.5"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7

    model_choice = st.sidebar.radio(
        "使用するモデルを選択:",
        model_options,
        index=model_options.index(st.session_state.model_choice),
        key="model_radio"
    )

    # モデル名マッピング
    model_name_map = {
        "GPT-3.5": "gpt-3.5-turbo",
        "GPT-4": "gpt-4o",
        "GPT-5": "gpt-5",
        "GPT-5 Mini": "gpt-5-mini",
        "Claude 3 Haiku": "claude-3-haiku-20240307",
        "Gemini 2.5 Pro": "gemini-2.5-pro",
        "Gemini 2.5 Flash": "gemini-2.5-flash",
    }

    st.session_state.model_choice = model_choice
    st.session_state.model_name = model_name_map[model_choice]

    # スライダー処理（GPT-4 でも確実に表示）
    if model_choice in ["GPT-5", "GPT-5 Mini"]:
        st.sidebar.info("⚠ GPT-5 系モデルは固定温度 1 のみ使用可能です。")
        temperature = 1.0
    elif "Claude" in model_choice:
        temperature = st.sidebar.slider(
            "温度 (創造性):", 0.0, 1.0, value=float(st.session_state.temperature), step=0.01, key="temp_slider"
        )
    else:
        temperature = st.sidebar.slider(
            "温度 (創造性):", 0.0, 2.0, value=float(st.session_state.temperature), step=0.01, key="temp_slider"
        )

    st.session_state.temperature = temperature

    # モデルインスタンス再生成
    try:
        if model_choice.startswith("GPT"):
            return ChatOpenAI(model_name=st.session_state.model_name, temperature=temperature)
        elif "Claude" in model_choice:
            return ChatAnthropic(model=st.session_state.model_name, temperature=temperature)
        else:
            return ChatGoogleGenerativeAI(model=st.session_state.model_name, temperature=temperature)
    except Exception as e:
        st.error(f"モデル初期化失敗: {e}")
        return None

# ==== トークン数計算 ====
def get_token_count(text, model_name):
    if "gemini" in model_name:
        return len(text) // 2
    try:
        encoding = tiktoken.encoding_for_model(model_name if "gpt" in model_name else "gpt-3.5-turbo")
        return len(encoding.encode(text))
    except KeyError:
        return len(text.split())

# ==== コスト計算 ====
def calc_and_display_costs():
    if "model_name" not in st.session_state:
        return
    input_count = output_count = 0
    for role, message in st.session_state.message_history:
        token_count = get_token_count(message, st.session_state.model_name)
        if role == "ai":
            output_count += token_count
        else:
            input_count += token_count
    if len(st.session_state.message_history) <= 1:
        return
    model = st.session_state.model_name
    input_cost = MODEL_PRICES["input"].get(model, 0) * input_count
    output_cost = MODEL_PRICES["output"].get(model, 0) * output_count
    total_cost = input_cost + output_cost
    st.sidebar.markdown("## 💰 コスト試算")
    st.sidebar.markdown(f"**合計コスト:** ${total_cost:.5f}")
    st.sidebar.markdown(f"- 入力コスト: ${input_cost:.5f}")
    st.sidebar.markdown(f"- 出力コスト: ${output_cost:.5f}")

# ==== メイン処理 ====
def main():
    init_page()
    init_messages()

    st.session_state.llm = select_model()

    for role, message in st.session_state.get("message_history", []):
        st.chat_message(role).markdown(message)

    user_input = st.chat_input("メッセージを入力してください...")
    if user_input and st.session_state.llm:
        st.chat_message("user").markdown(user_input)
        try:
            if "gemini" in st.session_state.model_name:
                response = st.session_state.llm.invoke([{"role": "user", "content": user_input}]).content
            elif "claude" in st.session_state.model_name:
                response = st.session_state.llm.invoke(user_input).content
            else:
                messages = [
                    HumanMessage(content=msg) if r == "user"
                    else AIMessage(content=msg) if r in ["assistant", "ai"]
                    else SystemMessage(content=msg)
                    for r, msg in st.session_state.message_history
                ]
                messages.append(HumanMessage(content=user_input))
                response = st.session_state.llm.invoke(messages).content
            st.chat_message("ai").markdown(response)
            st.session_state.message_history.append(("user", user_input))
            st.session_state.message_history.append(("ai", response))
        except Exception as e:
            st.error(f"LLM 応答生成エラー: {e}")

    calc_and_display_costs()

if __name__ == "__main__":
    main()