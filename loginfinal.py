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
from datetime import datetime
from supabase import create_client
from fpdf import FPDF


def get_base64_image(image_path):
    import base64
    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    return encoded


# ✅ Load environment variables
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
os.environ['TOGETHER_API_KEY'] = os.getenv("TOGETHER_API_KEY")
os.environ['COHERE_API_KEY'] = os.getenv("COHERE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


if "user_email" not in st.session_state:
    st.set_page_config(page_title="LegalAI Login", layout="centered")

    # Convert the image to base64
    bg_image = get_base64_image('/Users/bindudevidas/Desktop/Legal-CHATBOT-main/Untitled_design.jpg')

    # Embed the base64 image in CSS
    st.markdown(f"""
        <style>
            .stApp {{
                background-image: url(data:image/jpg;base64,{bg_image});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            .login-container {{
                background-color: rgba(255, 255, 255, 0.8);
                padding: 20px;
                border-radius: 10px;
                max-width: 400px;
                margin: auto;
                text-align: center;
            }}
        </style>
    """, unsafe_allow_html=True)

    # Login form
    st.title("🔐 LegalAI Login")
    email = st.text_input("📧 Email")
    password = st.text_input("🔑 Password", type="password")
    mode = st.radio("Select Mode", ["Login", "Signup"])
    if st.button("Continue"):
        try:
            if mode == "Login":
                supabase.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state.user_email = email
                st.success(f"✅ Logged in as {email}")
                st.rerun()
            else:
                supabase.auth.sign_up({"email": email, "password": password})
                st.success("📩 Signup successful! Please verify your email.")
        except Exception as e:
            st.error(f"❌ Auth failed: {e}")
    st.stop()



# ✅ Streamlit UI
st.set_page_config(page_title="LegalAI", layout="wide")
st.title("⚖️ LegalAI - Legal AI Chatbot")
st.sidebar.title("⚙️ Settings")
# ✅ Sidebar Options
response_length = st.sidebar.slider("Adjust Response Length:", 512, 4096, 2048, step=256)
model_choice = st.sidebar.selectbox("Choose an AI Model:", [
    "LLaMA 3 (Groq)", "Mixtral (Together AI)", "DeepSeek R1 (Groq)", "Command R (Cohere)"
])

# ✅ Memory State Setup
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=5, memory_key="chat_history", input_key="question", output_key="answer", return_messages=True
    )
if "current_section" not in st.session_state:
    st.session_state.current_section = None

# ✅ Load Vector Stores
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db_legal = FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True).as_retriever()
db_women = FAISS.load_local("women_laws_store", embeddings, allow_dangerous_deserialization=True).as_retriever()
db_schemes = FAISS.load_local("schemes_store", embeddings, allow_dangerous_deserialization=True).as_retriever()
db_bns = FAISS.load_local("bns_store", embeddings, allow_dangerous_deserialization=True).as_retriever()
db_student = FAISS.load_local("student_store", embeddings, allow_dangerous_deserialization=True).as_retriever()

# ✅ LLM Selection
llm = None
if model_choice == "LLaMA 3 (Groq)":
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", max_tokens=response_length)
elif model_choice == "Mixtral (Together AI)":
    llm = Together(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=os.getenv("TOGETHER_API_KEY"), max_tokens=response_length)
elif model_choice == "DeepSeek R1 (Groq)":
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="deepseek-r1-distill-llama-70b", max_tokens=response_length)
    print("✅ DeepSeek R1 (Groq) initialized successfully")
elif model_choice == "Command R (Cohere)":
    llm = ChatCohere(model="command-r", cohere_api_key=os.getenv("COHERE_API_KEY"), max_tokens=response_length)
# ✅ Keywords
legal_keywords = [
    "law", "legal", "ipc", "bns", "crpc", "constitution", "constitutional", "fundamental rights",
    "directive principles", "case", "cases", "trial", "litigation", "lawsuit", "section", "clause",
    "act", "statute", "code", "article", "jurisdiction", "court", "judge", "judgment", "judicial",
    "supreme court", "high court", "tribunal", "legal notice", "plaintiff", "defendant", "respondent",
    "petitioner", "petition", "affidavit", "summons", "appeal", "bail", "warrant", "arrest", 
    "evidence", "testimony", "hearing", "verdict", "chargesheet", "sentence", "punishment", 
    "crime", "criminal", "civil", "civil suit", "tort", "negligence", "contract", "agreement",
    "breach of contract", "property", "land", "ownership", "tenancy", "lease", "divorce", "marriage",
    "maintenance", "custody", "inheritance", "will", "succession", "adoption", "domestic violence",
    "sexual harassment", "human rights", "rights", "liberty", "freedom of speech", "right to equality",
    "right to education", "right to privacy", "right to life", "consumer protection", 
    "environmental law", "corporate law", "cyber law", "labour law", "employment law", "startup law", 
    "data protection", "information technology act", "rti", "intellectual property", "copyright", 
    "trademark", "patent", "gst", "tax law", "evidence act", "company law", "competition act",
    "arbitration", "conciliation", "mediation", "alternative dispute resolution", "nda", "internship",
    "student", "legal drafting", "legal research", "bar council", "advocate", "lawyer", "solicitor",
    "notary", "Bharatiya Nyaya Sanhita", "fir" , "rape", "student help", "law school", "moot court", "legal internship", "research paper", "legal notes",
    "academic project", "case study", "legal maxims", "case law", "jurisprudence", "legal research skills",
    "CV", "resume", "interview preparation", "legal career", "legal jobs", "junior advocate", "paralegal",
    "legal drafting skills", "internship guidance", "career guidance", "law firm", "law chambers",
    "legal consultant", "public defender", "legal aid", "pro bono", "lawyer directory" , " Economic and Social Council (ECOSOC)"
]

scheme_keywords = [
    "scheme", "yojana", "subsidy", "government scheme", "benefit", "welfare", "loan", 
    "pradhan", "mantri", "shram", "yogi", "maandhan", "han", "dhan", 
    "aicte", "rashtirya", "avishkar", "abhiyan", "bio", "energy", "gas", "capacity", 
    "building", "skill", "development", "comprehensive", "handicraft", "cluster", 
    "deen", "dayal", "disabled", "rehabilitation", "deendayal", "antyodaya", 
    "digital", "india", "land", "records", "modernisation", "implementation", 
    "imprint", "research", "initiative", "indira", "gandhi", "national", "disability", 
    "pension", "old", "age", "widow", "jan", "aushadhi", "kala", "sanskriti", "vikas", 
    "khadi", "gramodyog", "krishionnati", "livestock", "health", "disease", "control", 
    "msme", "champions", "afforestation", "apprenticeship", "training", "family", 
    "handloom", "mission", "rural", "drinking", "water", "employment", "guarantee", 
    "livelihood", "service", "urban", "new", "literacy", "north", "east", "special", 
    "infrastructure", "poshan", "adolescent", "girls", "creche", "ayushman", "bharat", 
    "health", "infrastructure", "garib", "kalyan", "anna", "arogya", "kisan", "samman", 
    "krishi", "sinchai", "prime", "minister", "employment", "generation", "gokul", 
    "uchhatar", "siksha", "reform", "linked", "distribution", "skill", "support", 
    "swachh", "unnat", "young", "leaders", 
]

# ✅ Section Selector
selected_section = st.sidebar.radio("📌 Choose a Section:", [
    "General Legal Chat", "Women's Laws & Rights", "Government Schemes", 
    "BNS 2023", "Student Help", "Find a Lawyer", "📜 Chat History"
])

# ✅ Section Switch Resets Chat
if selected_section != st.session_state.current_section:
    st.session_state.messages = []
    st.session_state.current_section = selected_section
# ✅ Chat History Page
if selected_section == "📜 Chat History":
    st.subheader("📜 Your Chat History")
    try:
        history = supabase.table("chat_history").select("*").eq("email", st.session_state.user_email).order("timestamp").execute().data
        if not history:
            st.info("No past chats found.")
        else:
            for item in history:
                role = "🧑 You" if item["role"] == "user" else "🤖 LegalAI"
                time = item["timestamp"][:19].replace("T", " ")
                st.markdown(f"**{role} @ {time}**  \n{item['content']}")
                st.markdown("---")
            if st.button("⬇️ Export Chat History as PDF"):
                try:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', size=16)
                    pdf.set_text_color(0, 51, 102)
                    pdf.cell(200, 10, txt="LegalAI Chat History", ln=True, align="C")
                    pdf.set_font("Arial", size=12)
                    pdf.ln(5)

                    for item in history:
                        role = "You" if item["role"] == "user" else "LegalAI"
                        time = item["timestamp"][:19].replace("T", " ")
                        text = f"{role} ({time}): {item['content']}"

                        # Set colors and borders based on the role
                        if role == "You":
                            pdf.set_fill_color(230, 240, 255)  # Light blue
                            pdf.set_text_color(0, 51, 153)  # Dark blue
                            border_style = 'LTRB'
                        else:
                            pdf.set_fill_color(240, 230, 255)  # Light purple
                            pdf.set_text_color(102, 0, 102)  # Dark purple
                            border_style = 'LTRB'

                        # Add rounded box with text
                        pdf.set_line_width(0.5)
                        pdf.multi_cell(0, 10, text.encode('latin-1', 'replace').decode('latin-1'), border=border_style, fill=True)
                        pdf.ln(2)  # Spacing between messages

                    pdf.output("styled_chat_history.pdf")
                    with open("styled_chat_history.pdf", "rb") as f:
                        st.download_button("📄 Download Styled PDF", f, file_name="LegalAI_Chat_History.pdf")
                    st.success("✅ PDF generated successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to generate PDF: {e}")
    except Exception as e:
        st.error(f"⚠️ Error retrieving chat history: {e}")
# ✅ Set Retriever
qa_retriever = None
if selected_section == "General Legal Chat":
    st.subheader("📖 General Legal Chat")
    qa_retriever = db_legal
elif selected_section == "Women's Laws & Rights":
    st.subheader("👩‍⚖️ Women's Laws & Rights")
    qa_retriever = db_women
elif selected_section == "Government Schemes":
    st.subheader("🏛️ Government Schemes")
    qa_retriever = db_schemes
elif selected_section == "BNS 2023":
    st.subheader("📘 Bharatiya Nyaya Sanhita (BNS) 2023")
    qa_retriever = db_bns
elif selected_section == "Student Help":
    st.subheader("🎓 Law Student Help")
    st.write("📚 Ask about law school, research, internships, etc.")
    qa_retriever = db_student
elif selected_section == "Find a Lawyer":
    st.subheader("⚖️ Find a Lawyer")
    st.info("🔎 Lawyer directory coming soon.")

# ✅ Setup QA Chain
qa = None
if qa_retriever:
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state.memory,
        retriever=qa_retriever,
        return_source_documents=True,
        output_key="answer"
    )

# ✅ Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ Chat Input Logic
input_prompt = st.chat_input("💬 Ask a legal question...")

if input_prompt and qa:
    with st.chat_message("user"):
        st.markdown(f"**You:** {input_prompt}")
    st.session_state.messages.append({"role": "user", "content": f"**You:** {input_prompt}"})

    is_legal = any(keyword in input_prompt.lower() for keyword in legal_keywords)
    is_scheme = any(keyword in input_prompt.lower() for keyword in scheme_keywords)

    allowed_sections = {
        "General Legal Chat": is_legal,
        "Women's Laws & Rights": is_legal,
        "BNS 2023": is_legal,
        "Student Help": is_legal,
        "Government Schemes": is_scheme,
        "Find a Lawyer": True
    }

    if not allowed_sections.get(selected_section, False):
        with st.chat_message("assistant"):
            st.error("🚫 Please ask a question relevant to the selected legal category.")
        st.session_state.messages.append({"role": "assistant", "content": "🚫 Only legal or scheme-related queries are allowed in this section."})
    else:
        with st.chat_message("assistant"):
            with st.spinner("📚 Processing..."):
                try:
                    response = qa.invoke({
                        "question": input_prompt,
                        "chat_history": st.session_state.memory.chat_memory
                    })

                    response_text = response.get("answer", "").strip()

                    if response_text and len(response_text) > 10:
                        st.markdown(f"**LegalAI:** {response_text}")
                        st.session_state.messages.append({"role": "assistant", "content": f"**LegalAI:** {response_text}"})
                    else:
                        fallback = llm.invoke(input_prompt).strip()
                        st.markdown(f"**LegalAI:** {fallback}")
                        st.session_state.messages.append({"role": "assistant", "content": f"**LegalAI:** {fallback}"})

                    now = datetime.utcnow().isoformat()
                    supabase.table("chat_history").insert({
                        "email": st.session_state.user_email,
                        "role": "user",
                        "content": input_prompt,
                        "timestamp": now
                    }).execute()
                    supabase.table("chat_history").insert({
                        "email": st.session_state.user_email,
                        "role": "assistant",
                        "content": response_text if response_text else fallback,
                        "timestamp": now
                    }).execute()

                except Exception as e:
                    st.error(f"⚠️ Error: {e}")
# ✅ Reset Chat
st.sidebar.button("🗑️ Reset Chat", on_click=lambda: st.session_state.clear())
if st.sidebar.button("🚪 Logout"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
