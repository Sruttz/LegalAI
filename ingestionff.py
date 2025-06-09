import os
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Please check your .env file.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Function to process and save documents into FAISS
def embed_and_save_documents(data_path, save_path):
    try:
        print(f"📁 Processing files from: {data_path}")

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        loader = PyPDFDirectoryLoader(data_path)
        docs = loader.load()

        if not docs:
            raise ValueError(f"No documents found in {data_path}. Ensure the folder contains PDFs.")

        print(f"✅ Loaded {len(docs)} documents from {data_path}.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)

        for doc in final_documents:
            doc.metadata["source"] = os.path.basename(doc.metadata.get("source", "unknown"))

        vector_store = FAISS.from_documents(final_documents, embeddings)
        vector_store.save_local(save_path)
        print(f"💾 Vector store saved at: {save_path}")

    except Exception as e:
        print(f"❌ Error: {e}")

# Create vector stores
if __name__ == "__main__":
    embed_and_save_documents("./LEGAL-DATA", "my_vector_store")
    embed_and_save_documents("./WOMEN-LAWS", "women_laws_store")
    embed_and_save_documents("./SCHEMES", "schemes_store")
    embed_and_save_documents("./BNS2023", "bns_store")           # New
    embed_and_save_documents("./STUDENT-HELP", "student_store")  # New
