from dotenv import load_dotenv
from pymongo.collection import Collection
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from Embedding_Store.Model import *
from Embedding_Store.chunking import *
from Embedding_Store.utils import *
from Embedding_Store.db import *
import os
load_dotenv()
# 1. Connect to MongoDB Atlas
try:
    mongodb_collection = connect_to_mongodb(os.getenv(
        "DATABASE_NAME"), os.getenv("COLLECTION_NAME"), os.getenv("CLUSTER_URL"))

    # (Optional) Clear the collection
    # print(f"Clearing existing documents from collection '{COLLECTION_NAME}'...")
    # delete_result = mongodb_collection.delete_many({})
    # print(f"Deleted {delete_result.deleted_count} documents.")

except Exception as e:
    print(f"Pipeline failed during MongoDB operations: {e}")
    exit()
# 2. Initialize Embedding Model
try:
    embeddings_model = initialize_embedding_model(
        os.getenv("MODEL_NAME_EMBED"))
except Exception:
    print("Pipeline stopped: Could not initialize embedding model.")
    exit()
# 3. Query Similar Vectors from MongoDB Atlas
vector_store = MongoDBAtlasVectorSearch(
    collection=mongodb_collection,
    embedding=embeddings_model,
    index_name=os.getenv("VECTOR_INDEX_NAME")
)

results = query_similar_vectors_from_mongodb(
    "What is JAVA?", vector_store, top_k=3)
print(results)
for doc in results:
    print(doc.page_content)
    print(doc.metadata)
