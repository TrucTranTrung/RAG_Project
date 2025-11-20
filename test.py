import time
from urllib.parse import quote_plus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_postgres import PGVector
import psycopg

def wait_for_db(host, port, user, password, dbname, retries=5, delay=3):
    """Đợi DB sẵn sàng trước khi connect"""
    for i in range(retries):
        try:
            conn = psycopg.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                dbname=dbname
            )
            conn.close()
            print("✅ Database is ready!")
            return True
        except Exception as e:
            print(f"Waiting for DB ({i+1}/{retries})... Error: {e}")
            time.sleep(delay)
    raise Exception("❌ Database not reachable after retries.")

def get_pgvector_store(collection_name: str = "Psychology") -> PGVector:
    db_user = "Rag_user"
    db_password = "Agents@123"
    db_name = "ragdatabase"
    db_host = "localhost"  # từ local
    db_port = 5432

    # Encode password nếu có ký tự đặc biệt
    encoded_password = quote_plus(db_password)
    connection_string = f"postgresql+psycopg://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"

    # Đợi DB ready
    wait_for_db(db_host, db_port, db_user, db_password, db_name)

    # Embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-base")

    # PGVector (extension phải đã có trong DB!)
    vector_store = PGVector(
        embeddings=embedding_model,
        collection_name=collection_name,
        connection=connection_string
    )
    print(f"✅ PGVector store '{collection_name}' ready.")
    return vector_store

if __name__ == "__main__":
    store = get_pgvector_store()
