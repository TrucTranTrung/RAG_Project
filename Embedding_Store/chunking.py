from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm.auto import tqdm
import torch 
from dotenv import load_dotenv

# __file__ là biến trỏ đến file Python hiện tại
script_directory = os.path.abspath(os.path.dirname(__file__))
print(f"Thư mục của script hiện tại: {script_directory}")
# os.path.dirname(script_directory) sẽ trả về đường dẫn của thư mục cha
project_root_directory = os.path.dirname(script_directory)
print(f"Thư mục gốc của dự án (dự kiến): {project_root_directory}")
# Tạo đường dẫn đầy đủ đến file .env trong thư mục 'config' của thư mục gốc dự án
dotenv_path = os.path.join(project_root_directory, 'config', '.env')

# Kiểm tra xem file .env có tồn tại không trước khi tải
if os.path.exists(dotenv_path):
    print(f"Đang tải biến môi trường từ: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
    print("Tải biến môi trường thành công.")
else:
    print(f"Cảnh báo: Không tìm thấy file .env tại {dotenv_path}")

similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD_FOR_MERGE"))
print(os.getenv('MODEL_NAME_EMBED'))
print(f"Similarity threshold: {similarity_threshold} (type: {type(similarity_threshold)})")
# ---------- STEP 1: Trích xuất text thường ----------
def split_documents_for_java(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    print(f"Splitting documents for Java with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}...")
    java_splitter = RecursiveCharacterTextSplitter.from_language(
        language="java",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_separator=True
    )
    initial_chunks = java_splitter.split_documents(documents)
    print(f"Initial splitting resulted in {len(initial_chunks)} chunks.")
    for i, chunk in enumerate(initial_chunks):
        if chunk.metadata is None:
            chunk.metadata = {}
        chunk.metadata['original_chunk_index_pass1'] = i
    return initial_chunks


def merge_chunks_by_semantic_similarity(
    initial_chunks: List[Document],
    langchain_embedding_model: HuggingFaceEmbeddings,
    similarity_threshold: float, # Type hint là float
    embedding_batch_size: int = 32  # Kích thước batch để tính embedding
) -> List[Document]:
    print(f"Starting semantic merging of {len(initial_chunks)} initial chunks with threshold {similarity_threshold}...")
    
    # Đảm bảo similarity_threshold là kiểu float
    try:
        current_similarity_threshold = float(similarity_threshold)
    except ValueError:
        print(f"Error: similarity_threshold '{similarity_threshold}' không thể chuyển đổi thành float. Sử dụng giá trị mặc định 0.8.")
        current_similarity_threshold = 0.8
    
    print(f"Using similarity threshold (float): {current_similarity_threshold}")

    if not initial_chunks:
        print("No initial chunks to process.")
        return []

    chunk_contents = [chunk.page_content for chunk in initial_chunks]
    num_chunks = len(chunk_contents)
    all_embeddings_list = []

    print(f"Calculating embeddings for {num_chunks} initial chunks in batches of {embedding_batch_size}...")
    try:
        for i in tqdm(range(0, num_chunks, embedding_batch_size), desc="Calculating Initial Embeddings"):
            batch_contents = chunk_contents[i:i + embedding_batch_size]
            if not batch_contents: # Bỏ qua nếu batch rỗng 
                continue
            
            # Tính embedding cho batch hiện tại
            batch_embeddings_list_of_lists = langchain_embedding_model.embed_documents(batch_contents)
            all_embeddings_list.extend(batch_embeddings_list_of_lists)
            
            # Giải phóng bộ nhớ cache CUDA sau mỗi batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if not all_embeddings_list:
            print("No embeddings were calculated. Cannot proceed with merging.")
            return []
            
        initial_embeddings = np.array(all_embeddings_list) # Chuyển sang numpy array để xử lý
        print(f"Calculated {len(initial_embeddings)} initial embeddings.")

    except Exception as e:
        print(f"Error embedding initial chunks with LangChain model: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Dọn dẹp nếu có lỗi
        return [] 

    merged_documents: List[Document] = []
    if initial_embeddings.size == 0:
        print("No embeddings to process for merging after calculation.")
        return merged_documents

    # --- Logic gộp chunk dựa trên độ tương đồng cosine ---
    current_merged_content_list = [initial_chunks[0].page_content]
    # Sao chép metadata và khởi tạo các trường metadata mới
    current_merged_metadata = initial_chunks[0].metadata.copy()
    current_merged_metadata['merged_from_original_indices'] = [current_merged_metadata.get('original_chunk_index', 0)] # Sử dụng key 
    current_merged_metadata['merged_content_sources'] = [current_merged_metadata.get('source', 'unknown')]


    for i in tqdm(range(1, len(initial_chunks)), desc="Merging Chunks by Similarity"):
        embedding_prev = initial_embeddings[i-1].reshape(1, -1) 
        embedding_curr = initial_embeddings[i].reshape(1, -1)

        try:
            # Đảm bảo similarity là Python float
            similarity_score = float(cosine_similarity(embedding_prev, embedding_curr)[0][0])
        except Exception as e_sim:
            print(f"Error calculating cosine similarity between chunk {i-1} and {i}: {e_sim}")
            similarity_score = 0.0 # Mặc định không gộp nếu có lỗi tính similarity

        # Thực hiện so sánh với current_similarity_threshold
        if similarity_score >= current_similarity_threshold:
            current_merged_content_list.append(initial_chunks[i].page_content)
            
            # Cập nhật metadata
            current_merged_metadata['merged_from_original_indices'].append(
                initial_chunks[i].metadata.get('original_chunk_index', i)
            )
            source_curr = initial_chunks[i].metadata.get('source', 'unknown')
            if source_curr not in current_merged_metadata['merged_content_sources']:
                current_merged_metadata['merged_content_sources'].append(source_curr)
        else:
            merged_page_content = " ".join(current_merged_content_list)
            # Đảm bảo sources là duy nhất và được sắp xếp
            current_merged_metadata['merged_content_sources'] = sorted(list(set(current_merged_metadata['merged_content_sources'])))
            merged_documents.append(Document(page_content=merged_page_content, metadata=current_merged_metadata))

            # Bắt đầu một chunk gộp mới với chunk hiện tại
            current_merged_content_list = [initial_chunks[i].page_content]
            current_merged_metadata = initial_chunks[i].metadata.copy()
            current_merged_metadata['merged_from_original_indices'] = [initial_chunks[i].metadata.get('original_chunk_index', i)]
            current_merged_metadata['merged_content_sources'] = [initial_chunks[i].metadata.get('source', 'unknown')]

    # Thêm chunk gộp cuối cùng còn lại 
    if current_merged_content_list:
        merged_page_content = " ".join(current_merged_content_list)
        current_merged_metadata['merged_content_sources'] = sorted(list(set(current_merged_metadata['merged_content_sources'])))
        merged_documents.append(Document(page_content=merged_page_content, metadata=current_merged_metadata))
    
    print(f"Semantic merging resulted in {len(merged_documents)} final chunks.")
    return merged_documents
