from pymongo import MongoClient
from pymongo.collection import Collection
from dotenv import load_dotenv
import getpass
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch


def get_mongodb_credentials():
    """Lấy username và password từ người dùng"""
    print("=== MongoDB Atlas Authentication ===")
    username = input("Nhập username: ")
    # getpass ẩn password khi nhập
    password = getpass.getpass("Nhập password: ")
    return username, password


def build_connection_string(username: str, password: str, cluster_url: str) -> str:
    """Tạo connection string từ username, password và cluster URL"""
    # Encode username và password để tránh các ký tự đặc biệt
    from urllib.parse import quote_plus

    encoded_username = quote_plus(username)
    encoded_password = quote_plus(password)

    connection_string = f"mongodb+srv://{encoded_username}:{encoded_password}@{cluster_url}/?retryWrites=true&w=majority&appName=Cluster0"
    return connection_string


def connect_to_mongodb(db_name: str, coll_name: str, cluster_url: str) -> Collection:
    """Kết nối đến MongoDB Atlas với username/password nhập từ bàn phím"""
    print(f"Chuẩn bị kết nối đến MongoDB Atlas...")

    try:
        # Lấy thông tin đăng nhập
        username, password = get_mongodb_credentials()

        # Tạo connection string
        connection_string = build_connection_string(
            username, password, cluster_url)

        print(f"Đang kết nối đến MongoDB Atlas...")
        # Kết nối đến MongoDB
        client = MongoClient(connection_string)

        # Test kết nối
        client.admin.command('ping')

        # Lấy database và collection
        db = client[db_name]
        collection = db[coll_name]

        print(f"✅ Kết nối thành công đến MongoDB Atlas!")
        print(f"   Database: '{db_name}'")
        print(f"   Collection: '{coll_name}'")

        return collection

    except Exception as e:
        print(f"❌ Lỗi kết nối đến MongoDB Atlas: {e}")
        print("Vui lòng kiểm tra lại username, password và kết nối mạng.")
        raise


def connect_to_mongodb_with_retry(db_name: str, coll_name: str, max_retries: int = 3) -> Collection:
    """Kết nối đến MongoDB với khả năng thử lại khi thất bại"""
    for attempt in range(max_retries):
        try:
            return connect_to_mongodb(db_name, coll_name)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Lần thử {attempt + 1} thất bại. Thử lại...")
                print("-" * 50)
            else:
                print(
                    f"Đã thử {max_retries} lần nhưng vẫn không kết nối được.")
                raise


def store_documents_in_mongodb_atlas(
    documents_to_store: List[Document],
    mongodb_collection: Collection,
    embedding_model: HuggingFaceEmbeddings,
    vector_index_name: str
):
    if not documents_to_store:
        print("No documents to store in MongoDB Atlas.")
        return
    print(f"Storing {len(documents_to_store)} documents in MongoDB Atlas collection '{mongodb_collection.name}' using index '{vector_index_name}'...")
    try:
        vector_store = MongoDBAtlasVectorSearch.from_documents(
            documents=documents_to_store,
            embedding=embedding_model,
            collection=mongodb_collection,
            index_name=vector_index_name
        )
        print(
            f"Successfully stored documents. Vector store ready: {vector_store is not None}")
    except Exception as e:
        print(f"Error storing documents in MongoDB Atlas Vector Search: {e}")
        print(
            f"Ensure vector index '{vector_index_name}' is correctly configured (dimensions, fields, similarity).")


def query_similar_vectors_from_mongodb(
    query: str,
    vector_store: MongoDBAtlasVectorSearch,
    top_k: int = 4
):
    """
    Truy vấn các vector/document tương tự nhất với câu hỏi từ vector db.
    Args:
        query (str): Câu hỏi truy vấn.
        vector_store (MongoDBAtlasVectorSearch): Instance đã khởi tạo từ collection, embedding, index_name.
        top_k (int): Số lượng kết quả trả về (mặc định 4).
    Returns:
        List[Document]: Danh sách các document tương tự nhất.
    """
    try:
        results = vector_store.similarity_search(query=query, k=top_k)
        return results
    except Exception as e:
        print(f"Lỗi khi truy vấn vector db: {e}")
        return []
