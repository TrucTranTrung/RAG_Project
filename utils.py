from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
import os

def load_pdf_documents(directory_path: str) -> List[Document]:
    print(f"Loading PDF documents from: {directory_path}...")
    if not os.path.exists(directory_path) or not os.listdir(directory_path): # Kiểm tra cả trường hợp thư mục rỗng
        print(f"Error: PDF directory '{directory_path}' does not exist or is empty.")
        print("Please create it and add your PDF files, or check the path.")
        return []
    loader = PyPDFDirectoryLoader(directory_path, recursive=True)
    try:
        documents = loader.load()
        if not documents:
            print("No documents were loaded. Check PDF files and loader permissions/path.")
            return []
        print(f"Successfully loaded {len(documents)} document pages.")
        return documents
    except Exception as e:
        print(f"Error loading PDF documents: {e}")
        return []









# Create for data chunking and passing through model 
data_chunk = [ ]