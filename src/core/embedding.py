from dotenv import load_dotenv
from chunking import DocumentProcessor
from utils import *
from db import *
# import torch
import torch
torch.cuda.empty_cache()
# --- MongoDB import ---

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
if os.path.exists(dotenv_path):
    print(f"Loading env file from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(f"warning: file .env at {dotenv_path} not found")

chunk_size = int(os.getenv("JAVA_SPLITTER_CHUNK_SIZE", "2000"))  
chunk_overlap = int(os.getenv("JAVA_SPLITTER_CHUNK_OVERLAP", "200"))

# 1. Load PDF Documents
raw_documents = DocumentProcessor.load_pdf_documents(os.getenv("PDF_DIRECTORY_PATH"))
if not raw_documents:
    print("Pipeline stopped: No documents loaded.")
    exit()

# 2. Initialize Embedding Model
try:
    embeddings_model = DocumentProcessor.initialize_embedding_model(
        os.getenv("MODEL_NAME_EMBED"))
except Exception:
    print("Pipeline stopped: Could not initialize embedding model.")
    exit()


# 3. Initial Splitting (Java Recursive)
print(chunk_size, chunk_overlap)
initial_chunks = DocumentProcessor.split_documents(
    raw_documents,
    chunk_size,
    chunk_overlap
)
if not initial_chunks:
    print("Pipeline stopped: No initial chunks created.")
    exit()
print(f"Initial chunks: {len(initial_chunks)}")

# 4. Custom Semantic Merging
final_merged_documents = DocumentProcessor.merge_chunks_by_semantic_similarity(
    initial_chunks,
    embeddings_model,
    os.getenv("SIMILARITY_THRESHOLD_FOR_MERGE")
)
if not final_merged_documents:
    print("Pipeline stopped: No documents after semantic merging.")
    exit()
    

try:
    collection_name = os.getenv("COLLECTION_NAME", "my_default_collection")

    # 1. Kết nối và lấy đối tượng vector store
    vector_store = get_pgvector_store(collection_name=collection_name)

    # 2. Lưu trữ các document đã xử lý vào PGVector
    store_documents_in_pgvector(
        final_merged_documents,
        vector_store
    )
    print("\nRAG data processing pipeline completed.")

except Exception as e:
    print(f"Pipeline failed during PGvector operations: {e}")
    exit() 


# 3. Truy vấn dữ liệu
# user_query = "What is Java and why should I use it?"
# similar_docs = query_similar_vectors_from_pgvector(user_query, vector_store, top_k=2)

# # 4. In kết quả
# print("\n--- KẾT QUẢ TRUY VẤN ---")
# output_path = "query_results.txt"
# with open(output_path, "w", encoding="utf-8") as f:
#     if similar_docs:
#         for doc, score in similar_docs:
#             print(f"Nội dung: {doc.page_content}")
#             print(f"Điểm tương đồng (score): {score:.4f}")
#             print("-" * 20)
#             # Write to file
#             f.write(f"Nội dung: {doc.page_content}\n")
#             f.write(f"Điểm tương đồng (score): {score:.4f}\n")
#             f.write("-" * 20 + "\n")
#     else:
#         print("Không tìm thấy document nào phù hợp.")
#         f.write("Không tìm thấy document nào phù hợp.\n")
# print(f"\nKết quả đã được lưu vào {output_path}")



