from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from API_Model.model import model_whisper 
import torch
import tempfile
import os


app = FastAPI()

@app.post("/answer")
async def rag_api(question: str = Form(None)):
    if not question:
        raise HTTPException(status_code=400, detail="You must provide a question.")

    if question:
        # xử lý câu hỏi văn bản
        return {"type": "text", "content": question}

    answer = generate_answer(question)
    return {
        "question": question,
        "answer": answer
    }


# if __name__ == "__main__":
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True
#     )