# pip install langchain-postgres langchain-huggingface python-dotenv psycopg-binary sentence-transformers
import os
from typing import List
import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from urllib.parse import quote_plus
from dotenv import load_dotenv
from tqdm.auto import tqdm

# __file__ là biến trỏ đến file Python hiện tại
script_directory = os.path.abspath(os.path.dirname(__file__))
project_root_directory = os.path.dirname(script_directory)
project_root_directory = os.path.dirname(project_root_directory)
dotenv_path = os.path.join(project_root_directory, 'config', '.env')

# Kiểm tra xem file .env có tồn tại không trước khi tải
if os.path.exists(dotenv_path):
    print(f"Đang tải biến môi trường từ: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(f"Cảnh báo: Không tìm thấy file .env tại {dotenv_path}")

def _embedding_device() -> str:
    configured_device = os.getenv("EMBEDDING_DEVICE", "auto").lower()
    if configured_device != "auto":
        return configured_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _embedding_batch_size(default: int = 8) -> int:
    try:
        return max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", str(default))))
    except ValueError:
        return default


def _pgvector_insert_batch_size(default: int = 8) -> int:
    try:
        return max(1, int(os.getenv("PGVECTOR_INSERT_BATCH_SIZE", str(default))))
    except ValueError:
        return default


embedding_device = _embedding_device()
embedding_batch_size = _embedding_batch_size()
embedding_model = HuggingFaceEmbeddings(
    model_name=os.getenv("MODEL_NAME_EMBED"),
    model_kwargs={"device": embedding_device},
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": embedding_batch_size,
    },
)
print(f"PGVector embedding model initialized on {embedding_device} with batch size {embedding_batch_size}.")

def get_pgvector_store(collection_name: str) -> PGVector:
    from urllib.parse import quote_plus
    import os

    db_user = os.getenv("POSTGRES_USER")
    db_pass = quote_plus(os.getenv("POSTGRES_PASSWORD", ""))
    db_name = os.getenv("POSTGRES_DB")
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "5432")

    if not all([db_user, db_pass, db_name]):
        raise ValueError("Missing POSTGRES_USER, POSTGRES_PASSWORD or POSTGRES_DB")

    conn_str = f"postgresql+psycopg://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    try:
        return PGVector(
            embeddings=embedding_model,
            collection_name=collection_name,
            connection=conn_str,
        )
    except Exception as e:
        print("❌ PGVector connection error:", e)
        raise


def store_documents_in_pgvector(
    documents_to_store: List[Document],
    vector_store: PGVector
):
    """
    Lưu trữ các document vào PGvector.
    """
    if not documents_to_store:
        print("Không có document nào để lưu trữ.")
        return

    collection_name = vector_store.collection_name
    print(f"Saving {len(documents_to_store)} document into collection '{collection_name}'...")
    
    insert_batch_size = _pgvector_insert_batch_size()
    try:
        for start in tqdm(
            range(0, len(documents_to_store), insert_batch_size),
            desc="Saving documents to PGVector",
        ):
            batch = documents_to_store[start:start + insert_batch_size]
            vector_store.add_documents(batch)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"✅ Successfully saved documents.")
    except Exception as e:
        print(f"❌ Error saving documents to PGvector: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


def query_similar_vectors_from_pgvector(
    query: str,
    vector_store: PGVector,
    top_k: int = 5
) -> List[Document]:
    """
    Truy vấn các document tương tự từ PGvector.
    """
    try:
        # Sử dụng similarity_search_with_score để tìm kiếm
        results_with_scores = vector_store.similarity_search_with_score(query=query, k=top_k)
        # print(f"✅ Tìm thấy {len(results_with_scores)} kết quả.")
        return results_with_scores
    except Exception as e:
        print(f"❌ Error querying vector: {e}")
        return []

