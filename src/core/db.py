# pip install langchain-postgres langchain-huggingface python-dotenv psycopg-binary sentence-transformers
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from urllib.parse import quote_plus
from dotenv import load_dotenv

# __file__ là biến trỏ đến file Python hiện tại
script_directory = os.path.abspath(os.path.dirname(__file__))
project_root_directory = os.path.dirname(script_directory)
project_root_directory = os.path.dirname(project_root_directory)
dotenv_path = os.path.join(project_root_directory, 'config', '.env')

# Kiểm tra xem file .env có tồn tại không trước khi tải
if os.path.exists(dotenv_path):
    print(f"Đang tải biến môi trường từ: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
    print("Tải biến môi trường thành công.")
else:
    print(f"Cảnh báo: Không tìm thấy file .env tại {dotenv_path}")

embedding_model = HuggingFaceEmbeddings(model_name=os.getenv("MODEL_NAME_EMBED"))
# print("Embedding model initialized.")

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


