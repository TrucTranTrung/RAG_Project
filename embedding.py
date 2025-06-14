from dotenv import load_dotenv
from pymongo.collection import Collection
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from Embedding_Store.Model import *
from Embedding_Store.chunking import *
from Embedding_Store.utils import *
from Embedding_Store.db import *
# LLM
from langchain_ollama import ChatOllama
import torch
print(torch.cuda.is_available())
# --- LangChain and related imports ---
# --- MongoDB import ---

# load env
basedir = os.path.abspath(os.path.dirname(__file__))
dotenv_path = os.path.join(basedir, 'config', '.env')
load_dotenv()
if os.path.exists(dotenv_path):
    print(f"Loading env file from: {dotenv_path}")
    # Tải các biến môi trường từ file .env được chỉ định
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(f"warning: file .env at {dotenv_path} not found")

chunk_size = int(os.getenv("JAVA_SPLITTER_CHUNK_SIZE", "1000")
                 )  # Default to 1000 if not set
# Default to 200 if not set
chunk_overlap = int(os.getenv("JAVA_SPLITTER_CHUNK_OVERLAP", "200"))


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
    embeddings_model = initialize_embedding_model(
        os.getenv("MODEL_NAME_EMBED"))
except Exception:
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
    embeddings_model,
    os.getenv("SIMILARITY_THRESHOLD_FOR_MERGE")
)
if not final_merged_documents:
    print("Pipeline stopped: No documents after semantic merging.")
    exit()

try:
    mongodb_collection = connect_to_mongodb(os.getenv(
        "DATABASE_NAME"), os.getenv("COLLECTION_NAME"), os.getenv("CLUSTER_URL"))

    # (Optional) Clear the collection
    # print(f"Clearing existing documents from collection '{COLLECTION_NAME}'...")
    # delete_result = mongodb_collection.delete_many({})
    # print(f"Deleted {delete_result.deleted_count} documents.")

    # 6. Store Final Merged Documents in MongoDB Atlas
    store_documents_in_mongodb_atlas(
        final_merged_documents,
        mongodb_collection,
        embeddings_model,
        os.getenv("VECTOR_INDEX_NAME")
    )
except Exception as e:
    print(f"Pipeline failed during MongoDB operations: {e}")
    exit()

print("\nRAG data processing pipeline completed.")


# similarities = model_embbed.similarity(embeddings, embeddings)
# print(embeddings.shape)
# dictionary
# key     values
# text : embedding

