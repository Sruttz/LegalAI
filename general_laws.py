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
st.set_page_config(page_title="📖 General Legal Chat", layout="wide")
st.title("📖 General Legal Chat")
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

# ✅ Load General Legal Vector Store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db_legal = FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True).as_retriever()

# ✅ LLM Selection
llm = None
if model_choice == "LLaMA 3 (Groq)":
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", max_tokens=response_length)
elif model_choice == "Mixtral (Together AI)":
    llm = Together(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=os.getenv("TOGETHER_API_KEY"), max_tokens=response_length)
elif model_choice == "Command R (Cohere)":
    llm = ChatCohere(model="command-r", cohere_api_key=os.getenv("COHERE_API_KEY"), max_tokens=response_length)

# ✅ General Legal Keywords
legal_keywords = [
    "law", "legal", "ipc", "bns", "crpc", "constitution", "case", "trial", "section", "clause", "act", "statute",
    "court", "judge", "judgment", "jurisdiction", "warrant", "arrest", "crime", "civil", "contract", "evidence",
    "maintenance", "custody", "divorce", "adoption", "rights", "liberty", "privacy", "property", "ownership",
    "notary", "advocate", "litigation", "plaintiff", "petition", "summons", "appeal", "sentence", "company law",
    "labour law", "cyber law", "consumer protection", "intellectual property", "tax law", "gst", "arbitration",
    "mediation", "nda", "legal drafting", "bar council", "supreme court", "high court", "Bharatiya Nyaya Sanhita"
]

# ✅ Setup QA Chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_legal,
    return_source_documents=True,
    output_key="answer"
)

# ✅ Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ Chat Input
input_prompt = st.chat_input("💬 Ask a legal question...")
if input_prompt:
    with st.chat_message("user"):
        st.markdown(f"**You:** {input_prompt}")
    st.session_state.messages.append({"role": "user", "content": f"**You:** {input_prompt}"})

    # ✅ Override for IPC–BNS confusion
    lower_input = input_prompt.lower()
    if ("ipc" in lower_input and "bns" in lower_input) or "ipc replaced" in lower_input:
        override_answer = (
            "**✅ Yes**, the Indian Penal Code (IPC) was officially **repealed** and replaced by the "
            "**Bharatiya Nyaya Sanhita (BNS), 2023**. It came into force on **July 1, 2024**, replacing the IPC "
            "as the primary penal law in India."
        )
        st.chat_message("assistant").markdown(f"**LawGPT:** {override_answer}")
        st.session_state.messages.append({"role": "assistant", "content": f"**LawGPT:** {override_answer}"})
    else:
        is_legal = any(keyword in lower_input for keyword in legal_keywords)

        if not is_legal:
            with st.chat_message("assistant"):
                st.error("🚫 Please ask a legal question relevant to Indian law.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "🚫 Only legal queries are allowed in this section."
            })
        else:
            with st.chat_message("assistant"):
                with st.status("📚 Processing...", expanded=True):
                    try:
                        response = qa.invoke({
                            "question": input_prompt,
                            "chat_history": st.session_state.memory.chat_memory
                        })

                        response_text = str(response.get("answer", "")).strip()
                        irrelevant_responses = [
                            "i don't know", "not found", "no relevant information", "text doesn't mention", "no details available"
                        ]

                        if not response_text or any(phrase in response_text.lower() for phrase in irrelevant_responses) or len(response_text) < 30:
                            fallback = llm.invoke(input_prompt)
                            fallback_text = str(fallback).strip()
                            st.markdown(f"**LawGPT (Fallback):** {fallback_text}")
                            st.session_state.messages.append({"role": "assistant", "content": f"**LawGPT (Fallback):** {fallback_text}"})
                        else:
                            st.markdown(f"**LawGPT:** {response_text}")
                            st.session_state.messages.append({"role": "assistant", "content": f"**LawGPT:** {response_text}"})
                    except Exception as e:
                        st.error(f"⚠️ Error: {e}")

# ✅ Reset Chat
st.sidebar.button("🗑️ Reset Chat", on_click=lambda: st.session_state.clear())
