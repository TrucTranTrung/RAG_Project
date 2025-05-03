from Model import model_embbed
from utils import data_chunk
from chunking import data_chunking
### LLM
from langchain_ollama import ChatOllama
import torch
print(torch.cuda.is_available())

# local_llm = "llama3.2:3b"
# llm = ChatOllama(model=local_llm, temperature=0)
# llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

# Dòng này sẽ gây lỗi nếu Ollama không chạy:
# response = llm.invoke("Xin chào!")
# print(response)
# chunking data
# should use button
data_chunking()
print(len(data_chunk))
my_dict = {}
i = 1
for chunk in data_chunk:
    embeddings = model_embbed.encode(chunk)
    my_dict[f'{chunk}']=embeddings
    # print(i)
    # i+=1
# Cho nay anh lay my_dict ra de lam viec
# anh sua file chunking.py de tra ve data_chunk
# print(len(my_dict))


# similarities = model_embbed.similarity(embeddings, embeddings)
# print(embeddings.shape)
# dictionary
# key     values
# text : embedding

