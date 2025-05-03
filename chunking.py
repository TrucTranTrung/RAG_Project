from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader
import os
from dotenv import load_dotenv, dotenv_values
from utils import data_chunk
# import argparse

# load env
basedir = os.path.abspath(os.path.dirname(__file__))
dotenv_path = os.path.join(basedir, 'config', '.env')

if os.path.exists(dotenv_path):
    print(f"Loading env file from: {dotenv_path}")
    # Tải các biến môi trường từ file .env được chỉ định
    load_dotenv(dotenv_path=dotenv_path) 
else:
    print(f"warning: file .env at {dotenv_path} not found")
    
print(os.getenv('model_name'))
# ---------- STEP 1: Trích xuất text thường ----------
def data_chunking():
    text_from_pdf = ""
    pdf_paths = []
    # Duyệt tất cả các file trong thư mục
    for root, dirs, files in os.walk(os.getenv('data_path')):
        for file in files:
            if file.endswith('.pdf'):  
                full_path = os.path.join(root, file)
                pdf_paths.append(full_path)


    for path in pdf_paths:
        with open(path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_from_pdf += page_text + "\n"


    # ---------- STEP 3: Semantic Chunking ----------
        embeddings = HuggingFaceEmbeddings(model_name=os.getenv('model_name'))
        chunker = SemanticChunker(embeddings)
        documents = chunker.create_documents([text_from_pdf])
        # print(documents)
        
    # Lưu tất cả các chunk vào một tệp
        namechunk = os.path.basename(path)
        # print(namechunk)
        with open(f"{os.getenv('output_chunk')}/{namechunk}_chunk.txt", "w") as f:
            for i, doc in enumerate(documents):
                f.write(f"--- Chunk #{i + 1} ---\n")
                f.write(doc.page_content + "\n")
                data_chunk.append(doc.page_content)
                # print(i)


# ---------- STEP 4: In ra vài chunks kiểm tra ----------
# for i, doc in enumerate(documents[:5]):  
#     print(f"\n--- Chunk #{i + 1} ---")
#     print(doc.page_content)
