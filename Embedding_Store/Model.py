from langchain_huggingface import HuggingFaceEmbeddings

def initialize_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    print(f"Initializing embedding model: {model_name}...")
    try:
        embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True,
                           'batch_size': 128}
        )
        print("Embedding model initialized.")
        return embeddings_model
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        raise
# model for chunking
