import os
import requests
import logging
from dotenv import load_dotenv
from typing import Union
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import JSONResponse
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import GoogleGenerativeAI
from langchain_community.vectorstores.pgvector import PGVector

from db import get_pgvector_store
from utils import get_top_k_contexts
from prompt import prompt_template

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI()

class RAGAPI:
    def __init__(self):
        self.collection_name = os.getenv("COLLECTION_NAME", "my_default_collection")
        self.llm = GoogleGenerativeAI(model="models/gemini-1.5-pro-latest")
        self.vector_store = get_pgvector_store(collection_name=self.collection_name)
        
        
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        # Xây dựng chuỗi LCEL.
        self.chain = (
            RunnableParallel(
                {
                    "context": self._get_reranked_contexts,
                    "question": RunnablePassthrough()
                }
            )
            | ChatPromptTemplate.from_template(prompt_template)
            | self.llm
            | StrOutputParser()
        )

    def _get_reranked_contexts(self, question: str):
        # Lấy tài liệu từ retriever.
        docs_with_scores = self.retriever.invoke(question)
        
        # Tách nội dung và điểm số.
        documents = [doc.page_content for doc in docs_with_scores]
        similarities = [doc.metadata.get("score", 0.0) for doc in docs_with_scores]
        
        # Áp dụng rerank.
        reranked_docs = get_top_k_contexts(documents, question, similarities, k=3)
        
        # Trả về chuỗi tài liệu đã được định dạng.
        return "\n\n".join(reranked_docs)

    def handle_question(self, question: str):
        output_text = self.chain.invoke(question)
        return output_text

    def handle_audio(self, audio: UploadFile):
        try:
            response_stt = requests.post(
                "http://whisper-api:8000/STT/",
                files={"file": audio.file}
            )
            response_stt.raise_for_status()

            data = response_stt.json()
            output_stt = data["output_text"]

            output_text = self.handle_question(output_stt)

            answer_audio_base64 = requests.post(
                "http://tts-api:8001/transcribe/",
                data={"text_input": output_text}
            )
            answer_audio_base64.raise_for_status()

            tts_data = answer_audio_base64.json()
            final_audio_base64 = tts_data.get("output_sound")

            return {"type": "audio", "audio_base64": final_audio_base64}

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request to another service failed: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to communicate with an internal service."}
            )
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": "An internal server error occurred."}
            )

rag_api_service = RAGAPI()

@app.post("/answer")
async def rag_api(question: str = Form(None), audio: Union[UploadFile, str] = File(None)):
    if isinstance(audio, str) and audio == "":
        audio = None

    if not question and not audio:
        return JSONResponse(
            status_code=400,
            content={"error": "No question or audio provided"}
        )

    if question:
        output_text = rag_api_service.handle_question(question)
        return {"type": "text", "content": output_text}
    else:
        return rag_api_service.handle_audio(audio)

# uvicorn API_LLAMA3_2:app --host 0.0.0.0 --port 4096 --reload



# convert speech to text
# response_stt = requests.post("http://0.0.0.0:8000/STT/", files={"file": audio.file})
# data = response_stt.json()
# output_stt = data["output_text"]
# # --- Query PGVector ---
# output_database = query_similar_vectors_from_pgvector(output_stt, vector_store, top_k=5)

# # rerank contexts
# similarities = []
# documents = []
# for document, score in output_database:
#     documents.append(document.page_content)
#     similarities.append(score)
# reranked_indices = get_top_k_contexts(documents, output_stt, similarities, k=3)

# output_text = get_entities_as_string_GEMINI(prompt_template, information=reranked_indices, question=output_stt)
# # convert text to speech
# answer_audio_base64 = requests.post("http://0.0.0.0:8001/transcribe/", data={"text_input": output_text})
# answer_audio_base64.raise_for_status() # HTTP error

# # Extract JSON from response
# tts_data = answer_audio_base64.json()
# final_audio_base64 = tts_data.get("output_sound")

# return {"type": "audio","audio_base64": final_audio_base64}