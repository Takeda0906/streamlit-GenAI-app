import os
import streamlit as st
import tiktoken
from langchain.chat_models.openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# ==== 安全に LangSmith を初期化 ====
def setup_langsmith():
    try:
        from langsmith import Client

        api_key = os.getenv("LANGCHAIN_API_KEY")
        tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower() in ["true", "1", "yes"]

        st.sidebar.markdown("## 🧠 LangSmith ログ設定")
        st.sidebar.write("LANGCHAIN_API_KEY:", api_key)
        st.sidebar.write("LANGCHAIN_TRACING_V2:", os.getenv("LANGCHAIN_TRACING_V2"))
        st.sidebar.write("tracing_enabled:", tracing_enabled)

        if not api_key:
            st.sidebar.warning("⚠ LangSmith APIキーが未設定です。")
            return None

        client = Client(api_key=api_key)

        if tracing_enabled:
            st.sidebar.success("✅ LangSmith ログ送信が有効です")
        else:
            st.sidebar.warning("⚠ LangSmith ログ送信は無効です（LANGCHAIN_TRACING_V2 を確認）")

        return client

    except Exception as e:
        st.sidebar.error(f"LangSmith 初期化エラー: {e}")
        return None


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
        temperature = st.sidebar.slider("温度 (創造性):", 0.0, 1.0, default_temp, 0.01, key="temp_claude")
    else:
        temperature = st.sidebar.slider("温度 (創造性):", 0.0, 2.0, default_temp, 0.01, key="temp_other")

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
        return len(encoding.encode(text))


# ==== コスト試算 ====
def calc_and_display_costs():
    if "model_name" not in st.session_state:
        return
    input_count = 0
    output_count = 0
    for role, message in st.session_state.message_history:
        token_count = get_token_count(message, st.session_state.model_name)
        if role == "ai":
            output_count += token_count
        else:
            input_count += token_count
    model = st.session_state.model_name
    input_cost = MODEL_PRICES["input"].get(model, 0) * input_count
    output_cost = MODEL_PRICES["output"].get(model, 0) * output_count
    total = input_cost + output_cost
    st.sidebar.markdown("## 💰 コスト試算")
    st.sidebar.markdown(f"**合計:** ${total:.5f}")
    st.sidebar.markdown(f"- 入力: ${input_cost:.5f}")
    st.sidebar.markdown(f"- 出力: ${output_cost:.5f}")


# ==== メイン ====
def main():
    init_page()
    langsmith_client = setup_langsmith()
    init_messages()
    select_model()

    if "llm" not in st.session_state or st.session_state.llm is None:
        st.warning("モデルが初期化されていません。")
        return

    for role, message in st.session_state.get("message_history", []):
        st.chat_message(role).markdown(message)

    user_input = st.chat_input("メッセージを入力してください...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        try:
            if "gemini" in st.session_state.model_name:
                response = st.session_state.llm.invoke([{"role": "user", "content": user_input}]).content
            elif "claude" in st.session_state.model_name:
                response = st.session_state.llm.invoke(user_input).content
            else:
                messages_for_gpt = [
                    HumanMessage(content=c)
                    if r == "user"
                    else AIMessage(content=c)
                    if r in ["assistant", "ai"]
                    else SystemMessage(content=c)
                    for r, c in st.session_state.message_history
                ]
                messages_for_gpt.append(HumanMessage(content=user_input))
                response = st.session_state.llm.invoke(messages_for_gpt).content

            st.chat_message("ai").markdown(response)
            st.session_state.message_history.extend([("user", user_input), ("ai", response)])

            # LangSmith に安全にログ送信（run_type="llm"）
            if langsmith_client:
                try:
                    run = langsmith_client.create_run(
                        run_type="llm",
                        name=f"Chat - {st.session_state.model_name}",
                        inputs={"prompt": user_input},
                        outputs={"response": response},
                        tags=["streamlit", st.session_state.model_name],
                    )
                    st.sidebar.success(f"✅ Run 作成成功: {run.id}")
                except Exception as log_err:
                    st.sidebar.error(f"❌ Run 作成失敗: {log_err}")

        except Exception as e:
            st.error(f"応答生成エラー: {e}")

    calc_and_display_costs()


if __name__ == "__main__":
    main()

