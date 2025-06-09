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
st.set_page_config(page_title="🎓 Student Help", layout="wide")
st.title("🎓 Law Student Help")
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

# ✅ Load Student Vector Store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db_student = FAISS.load_local("student_store", embeddings, allow_dangerous_deserialization=True).as_retriever()

# ✅ LLM Selection
llm = None
if model_choice == "LLaMA 3 (Groq)":
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", max_tokens=response_length)
elif model_choice == "Mixtral (Together AI)":
    llm = Together(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=os.getenv("TOGETHER_API_KEY"), max_tokens=response_length)
elif model_choice == "Command R (Cohere)":
    llm = ChatCohere(model="command-r", cohere_api_key=os.getenv("COHERE_API_KEY"), max_tokens=response_length)

# ✅ Student Help Keywords
student_keywords = [
    "law school", "law student", "llb", "llm", "clat", "judiciary exam", "moot court", "legal drafting", "drafting",
    "contract drafting", "case brief", "case analysis", "bare act", "legal research", "legal writing", "legal essay",
    "internship", "legal internship", "internship certificate", "vacation scheme", "training contract", "law firm",
    "resume", "cv", "cover letter", "sop", "statement of purpose", "personal statement", "portfolio",
    "assignment", "research paper", "term paper", "dissertation", "thesis", "plagiarism", "turnitin", "citation",
    "bluebook", "oscola", "apa", "mla", "footnotes", "endnotes", "presentation", "viva", "exam", "internal marks",
    "coursework", "module", "syllabus", "subject", "semester", "credit", "class test", "academic year",
    "recruitment", "placement", "campus drive", "interview", "mock interview", "career", "career guidance",
    "specialization", "criminal law", "constitutional law", "civil law", "contract law", "family law", "property law",
    "labour law", "human rights law", "environmental law", "cyber law", "intellectual property", "company law",
    "tax law", "startup law", "jurisprudence", "torts", "ipc", "crpc", "evidence act", "bns", "civil procedure code",
    "public international law", "private international law", "legal aid", "legal clinic", "law review", "student journal",
    "conference", "workshop", "seminar", "law fest", "law conclave", "client counselling", "negotiation", "mediation",
    "arbitration", "memorandum", "plaint note", "written submission", "model united nations", "student body", "debate",
    "quiz", "legal awareness", "mock trial", "advocacy skills", "court visit", "judgment writing", "tribunal", "legal blog",
    "contract", "agreement", "legal agreement", "tenant", "landlord", "lease", "rent", "notice", "affidavit", "legal notice"
]


# ✅ Setup QA Chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_student,
    return_source_documents=True,
    output_key="answer"
)

# ✅ Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ Chat Input Logic
input_prompt = st.chat_input("💬 Ask your student-related legal question...")
if input_prompt:
    with st.chat_message("user"):
        st.markdown(f"**You:** {input_prompt}")
    st.session_state.messages.append({"role": "user", "content": f"**You:** {input_prompt}"})

    is_student = any(keyword in input_prompt.lower() for keyword in student_keywords)

    if not is_student:
        with st.chat_message("assistant"):
            st.error("🚫 Please ask a law student-related or academic legal question.")
        st.session_state.messages.append({"role": "assistant", "content": "🚫 Only law student or academic queries are allowed in this section."})
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
