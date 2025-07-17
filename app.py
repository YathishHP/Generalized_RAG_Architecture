import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

# ------------------------
# Custom CSS styling
# ------------------------
st.markdown("""
<style>
  /* Main background & text */
  .main {
      background-color: #1a1a1a;
      color: #ffffff;
  }
  .sidebar .sidebar-content {
      background-color: #2d2d2d;
  }
  .stTextInput textarea {
      color: #ffffff !important;
  }
  /* Select box styling */
  .stSelectbox div[data-baseweb="select"] {
      color: white !important;
      background-color: #3d3d3d !important;
  }
  .stSelectbox svg {
      fill: white !important;
  }
  .stSelectbox option {
      background-color: #2d2d2d !important;
      color: white !important;
  }
  div[role="listbox"] div {
      background-color: #2d2d2d !important;
      color: white !important;
  }

  /* ----- Model Capabilities List Styling ----- */
  .sidebar .stMarkdown ul li {
    list-style: none;
    background-color: #2A2A2A !important;
    border-radius: 6px;
    padding: 6px 10px;
    margin: 4px 0;
    display: flex;
    align-items: center;
    color: #F0F0F0 !important;
  }
  .sidebar .stMarkdown ul li::before {
    content: "";
    display: inline-block;
    width: 1.5em;
    text-align: center;
    margin-right: 8px;
  }
  .sidebar .stMarkdown ul li span.emoji {
    color: #00FFAA;
    display: inline-block;
    margin-right: 4px;
  }
</style>
""", unsafe_allow_html=True)

st.title("🧩 QuadCore Code Ally")
st.caption("Four-mode AI: Python Expert · Debugging Assistant · Code Documentation · Solution Architect")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# ------------------------
# Sidebar configuration
# ------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["Gemma 2B", "DeepSeek Chat", "Mistral Instruct"],
        index=0
    )
    mode = st.selectbox(
        "Select Assistant Mode",
        ["Python Expert", "Debugging Assistant", "Code Documentation", "Solution Design"],
        index=0
    )
    st.divider()
    st.sidebar.markdown("""
    ### Model Capabilities
    <ul>
      <li><span class="emoji">🐍</span> Python Expert</li>
      <li><span class="emoji">🐞</span> Debugging Assistant</li>
      <li><span class="emoji">📝</span> Code Documentation</li>
      <li><span class="emoji">💡</span> Solution Design</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("Built with Streamlit, LangChain & OpenRouter")

# ------------------------
# Model initialization
# ------------------------
model_map = {
    "Gemma 2B": "google/gemma-2-9b-it:free",
    "DeepSeek Chat": "deepseek/deepseek-chat-v3-0324:free",
    "Mistral Instruct": "cognitivecomputations/dolphin3.0-mistral-24b:free"
}

LANGUAGE_MODEL = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model=model_map[selected_model],
    temperature=0.3
)

# ------------------------
# Dynamic prompt template
# ------------------------
PROMPT_TEMPLATE = """
You are a {mode}.
Provide your response in English.
Be concise, correct, and—where applicable—show code examples or debugging print statements.

Question:
{question}
"""

def generate_ai_response(mode: str, question: str) -> str:
    prompt_chain = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    pipeline = prompt_chain | LANGUAGE_MODEL | StrOutputParser()
    return pipeline.invoke({"mode": mode, "question": question})

# ------------------------
# Session state & chat UI
# ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}
    ]

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_query = st.chat_input("Type your question here...")

if user_query:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Generate AI response
    with st.spinner("🧠 Thinking..."):
        ai_reply = generate_ai_response(mode, user_query)
    st.session_state.messages.append({"role": "ai", "content": ai_reply})

    # Refresh UI
    st.rerun()
