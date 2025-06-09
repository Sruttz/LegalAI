from PyPDF2 import PdfReader
import os

folder_path = "./"

for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        try:
            reader = PdfReader(os.path.join(folder_path, filename))
            _ = reader.pages[0]  # Try reading first page
        except Exception as e:
            print(f"❌ Corrupted: {filename} — {e}")
