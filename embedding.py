from Model import model_embbed
from utils import sentences
from chunking import data_chunking
### LLM
from langchain_ollama import ChatOllama

local_llm = "llama3.2:3b"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

# Dòng này sẽ gây lỗi nếu Ollama không chạy:
# response = llm.invoke("Xin chào!")
# print(response)
# chunking data
# should use button
data_chunking()

# embeddings = model_embbed.encode(sentences)

# similarities = model_embbed.similarity(embeddings, embeddings)
# print(embeddings.shape)


