# 📁 FILE: women_laws_rights.py

import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_together import Together
from langchain_cohere import ChatCohere
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()
os.environ['TOGETHER_API_KEY'] = os.getenv("TOGETHER_API_KEY")
os.environ['COHERE_API_KEY'] = os.getenv("COHERE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# ✅ Streamlit UI
st.set_page_config(page_title="👩‍⚖️ Women's Laws & Rights", layout="wide")
st.title("👩‍⚖️ Women's Laws & Rights")
st.sidebar.title("⚙️ Settings")

# ✅ Sidebar Options
response_length = st.sidebar.slider("Adjust Response Length:", 512, 4096, 2048, step=256)
model_choice = st.sidebar.selectbox("Choose an AI Model:", [
    "LLaMA 3 (Groq)", "Mixtral (Together AI)", "Command R (Cohere)"
])

# ✅ Memory State Setup
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=5, memory_key="chat_history", input_key="question", output_key="answer", return_messages=True
    )

# ✅ Load Vector Store for Women’s Laws
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db_women = FAISS.load_local("women_laws_store", embeddings, allow_dangerous_deserialization=True).as_retriever()

# ✅ LLM Selection
llm = None
if model_choice == "LLaMA 3 (Groq)":
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", max_tokens=response_length)
elif model_choice == "Mixtral (Together AI)":
    llm = Together(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=os.getenv("TOGETHER_API_KEY"), max_tokens=response_length)
elif model_choice == "Command R (Cohere)":
    llm = ChatCohere(model="command-r", cohere_api_key=os.getenv("COHERE_API_KEY"), max_tokens=response_length)

# ✅ Keywords
legal_keywords = [
    "law", "legal", "domestic violence", "sexual harassment", "marriage", "divorce", "maintenance", "custody",
    "women rights", "dowry", "molestation", "workplace harassment", "POSH", "marital rape", "maternity", "gender equality",
    "fundamental rights", "legal aid", "crime against women", "protection of women", "indecent representation", "section 498a",
    "prohibition of child marriage", "female foeticide", "acid attack", "sexual assault", "rights of women", "women laws"
]

# ✅ Setup QA Chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_women,
    return_source_documents=True,
    output_key="answer"
)

# ✅ Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ Chat Input Logic
input_prompt = st.chat_input("💬 Ask about women’s legal rights...")
if input_prompt:
    with st.chat_message("user"):
        st.markdown(f"**You:** {input_prompt}")
    st.session_state.messages.append({"role": "user", "content": f"**You:** {input_prompt}"})

    is_legal = any(keyword in input_prompt.lower() for keyword in legal_keywords)

    if not is_legal:
        with st.chat_message("assistant"):
            st.error("🚫 Please ask a question related to women's legal rights.")
        st.session_state.messages.append({"role": "assistant", "content": "🚫 Only questions about women's laws and rights are allowed in this section."})
    else:
        with st.chat_message("assistant"):
            with st.status("📚 Processing...", expanded=True):
                try:
                    response = qa.invoke({"question": input_prompt, "chat_history": st.session_state.memory.chat_memory})
                    response_text = response.get("answer", "").strip()

                    if response_text and len(response_text) > 10:
                        st.markdown(f"**LawGPT:** {response_text}")
                        st.session_state.messages.append({"role": "assistant", "content": f"**LawGPT:** {response_text}"})
                    else:
                        fallback = llm.invoke(input_prompt).strip()
                        st.markdown(f"**LawGPT:** {fallback}")
                        st.session_state.messages.append({"role": "assistant", "content": f"**LawGPT:** {fallback}"})
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

# ✅ Reset Chat
st.sidebar.button("🗑️ Reset Chat", on_click=lambda: st.session_state.clear())
