from Model import *
from utils import *
from chunking import *
### LLM
from langchain_ollama import ChatOllama
import torch
print(torch.cuda.is_available())
# --- LangChain and related imports ---
from langchain_mongodb import MongoDBAtlasVectorSearch
# --- MongoDB import ---
from pymongo import MongoClient
from pymongo.collection import Collection
from dotenv import load_dotenv

# load env
basedir = os.path.abspath(os.path.dirname(__file__))
dotenv_path = os.path.join(basedir, 'config', '.env')

if os.path.exists(dotenv_path):
    print(f"Loading env file from: {dotenv_path}")
    # Tải các biến môi trường từ file .env được chỉ định
    load_dotenv(dotenv_path=dotenv_path) 
else:
    print(f"warning: file .env at {dotenv_path} not found")
    
chunk_size = int(os.getenv("JAVA_SPLITTER_CHUNK_SIZE", "1000"))  # Default to 1000 if not set
chunk_overlap = int(os.getenv("JAVA_SPLITTER_CHUNK_OVERLAP", "200"))  # Default to 200 if not set


# import argparse
# local_llm = "llama3.2:3b"
# llm = ChatOllama(model=local_llm, temperature=0)
# llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

# Dòng này sẽ gây lỗi nếu Ollama không chạy:
# response = llm.invoke("Xin chào!")
# print(response)
# chunking data
# should use button
# 1. Load PDF Documents
raw_documents = load_pdf_documents(os.getenv("PDF_DIRECTORY_PATH"))
if not raw_documents:
    print("Pipeline stopped: No documents loaded.")
    exit()

# 2. Initialize Embedding Model
try:
    embeddings_model = initialize_embedding_model(os.getenv("MODEL_NAME_EMBED"))
except Exception: # Bắt lỗi nếu không khởi tạo được model
    print("Pipeline stopped: Could not initialize embedding model.")
    exit()


# 3. Initial Splitting (Java Recursive)
initial_chunks = split_documents_for_java(
    raw_documents,
    chunk_size,
    chunk_overlap
)
if not initial_chunks:
    print("Pipeline stopped: No initial chunks created.")
    exit()
print(f"Initial chunks: {len(initial_chunks)}")

# 4. Custom Semantic Merging
final_merged_documents = merge_chunks_by_semantic_similarity(
    initial_chunks,
    embeddings_model, # Thay đổi ở đây
    os.getenv("SIMILARITY_THRESHOLD_FOR_MERGE")
)
if not final_merged_documents:
    print("Pipeline stopped: No documents after semantic merging.")
    exit()
    
    
# Cho nay anh lay my_dict ra de lam viec
# anh sua file chunking.py de tra ve data_chunk
# print(len(my_dict))


# similarities = model_embbed.similarity(embeddings, embeddings)
# print(embeddings.shape)
# dictionary
# key     values
# text : embedding

