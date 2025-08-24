# pip install langchain-postgres langchain-huggingface python-dotenv psycopg-binary sentence-transformers
import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()


def get_pgvector_store(collection_name: str) -> PGVector:
    """
    Kết nối đến PGvector và trả về một đối tượng vector store.
    """
    print("Connecting to PGvector...")

    # Lấy thông tin từ các biến môi trường
    db_user = os.getenv("POSTGRES_USER")
    db_password = quote_plus(os.getenv("POSTGRES_PASSWORD"))
    db_name = os.getenv("POSTGRES_DB")
    db_host = os.getenv("POSTGRES_CONTAINER_HOST")
    db_port = os.getenv("POSTGRES_PORT")

    if not all([db_user, db_password, db_name]):
        raise ValueError("Pls provide POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB in .env")

    # Tạo connection string cho PostgreSQL
    connection_string = f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    print(f"Connecting to PGvector with database '{db_name}'...")
    # print(f"Connecting to PGvector with database '{db_password}'...")
    # print(f"Connecting to PGvector with database '{db_host}'...")

    try:
        embedding_model = HuggingFaceEmbeddings(model_name=os.getenv("MODEL_NAME_EMBED"))

        # Khởi tạo đối tượng PGVector
        vector_store = PGVector(
            embeddings=embedding_model,
            collection_name=collection_name,
            connection=connection_string,
        )
        print("✅ Connection and PGVector store initialization successful!")
        return vector_store

    except Exception as e:
        print(f"❌ Error connecting to PGvector: {e}")
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
    
    try:
        # Hàm add_documents sẽ thêm các document vào database
        vector_store.add_documents(documents_to_store)
        print(f"✅ Successfully saved documents.")
    except Exception as e:
        print(f"❌ Error saving documents to PGvector: {e}")


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


