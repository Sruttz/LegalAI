import streamlit as st
import os
import time
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_together import Together
from langchain_cohere import ChatCohere
from langchain_openai import OpenAI  # Used for OpenRouter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['TOGETHER_API_KEY'] = os.getenv("TOGETHER_API_KEY")
os.environ['MISTRAL_API_KEY'] = os.getenv("MISTRAL_API_KEY")
os.environ['COHERE_API_KEY'] = os.getenv("COHERE_API_KEY")
os.environ['OPENROUTER_API_KEY'] = os.getenv("OPENROUTER_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="LawGPT")

col1, col2, col3 = st.columns([1, 4, 1])
st.title("Llama Model Legal ChatBot")

st.sidebar.title("Settings ⚙️")
model_choice = st.sidebar.selectbox(
    "Choose a Free AI Model:",
    ["LLaMA 3 (Groq)", "Mixtral (Together AI)", "Mistral-7B", "Command R (Cohere)", "OpenRouter (Mixtral)"]
)

# ✅ Check if API keys are available before initializing models
if not groq_api_key and model_choice == "LLaMA 3 (Groq)":
    st.error("❌ Groq API Key is missing! Check your .env file.")
if not os.getenv("TOGETHER_API_KEY") and model_choice == "Mixtral (Together AI)":
    st.error("❌ Together AI API Key is missing! Check your .env file.")
if not os.getenv("MISTRAL_API_KEY") and model_choice == "Mistral-7B":
    st.error("❌ Mistral API Key is missing! Check your .env file.")
if not os.getenv("COHERE_API_KEY") and model_choice == "Command R (Cohere)":
    st.error("❌ Cohere API Key is missing! Check your .env file.")
if not os.getenv("OPENROUTER_API_KEY") and model_choice == "OpenRouter (Mixtral)":
    st.error("❌ OpenRouter API Key is missing! Check your .env file.")

# Reset conversation function
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# ✅ Initialize Embeddings (Fix for FAISS Error)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# ✅ Load FAISS Vector Store (Fixed)
db = FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Define the prompt template
prompt_template = """
<s>[INST]This is a chat template and As a legal chat bot, your primary objective is to provide accurate and concise information based on the user's questions. 
Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details.
Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, 
you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response.
You will prioritize the user's query and refrain from posing additional questions. 
The aim is to deliver professional, precise, and contextually relevant information pertaining to the Indian Penal Code.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

# ✅ Handle model initialization properly
try:
    if model_choice == "LLaMA 3 (Groq)":
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    elif model_choice == "Mixtral (Together AI)":
        llm = Together(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=os.getenv("TOGETHER_API_KEY"))
    elif model_choice == "Mistral-7B":
        llm = ChatMistralAI(model="mistral-7b-instruct", api_key=os.getenv("MISTRAL_API_KEY"))
    elif model_choice == "Command R (Cohere)":
        llm = ChatCohere(model="command-r", cohere_api_key=os.getenv("COHERE_API_KEY"))
    elif model_choice == "OpenRouter (Mixtral)":
        llm = OpenAI(
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
            base_url="https://openrouter.ai/api/v1"
        )
    else:
        st.error("⚠️ No model selected!")
        llm = None
except Exception as e:
    st.error(f"⚠️ Error initializing model: {e}")
    llm = None

# ✅ Only create the QA chain if the model is initialized
if llm:
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state.memory,
        retriever=db_retriever,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

# Input prompt
input_prompt = st.chat_input("Ask me a legal question...")

if input_prompt and llm:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role": "user", "content": input_prompt})

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...", expanded=True):
            try:
                result = qa.invoke(input=input_prompt)
                message_placeholder = st.empty()
                full_response = "\n\n\n"

                for chunk in result["answer"]:
                    full_response += chunk
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + " ▌")

            except Exception as e:
                st.error(f"⚠️ Error processing request: {e}")

        st.button('Reset All Chat 🗑️', on_click=reset_conversation)
    
    st.session_state.messages.append({"role": "assistant", "content": result["answer"] if 'result' in locals() else "⚠️ Unable to process request."})
