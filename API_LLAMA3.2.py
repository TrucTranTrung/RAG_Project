from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from API_Model.model import model_whisper 
import torch
import tempfile
import os


app = FastAPI()

# Load mô hình SPEECH TO TEXT
model_whisper = model_whisper

def transcribe_audio(file: UploadFile) -> str:
    segments, info = model_whisper.transcribe("/home/daniel/Documents/RAG_Java/harvard.wav", beam_size=5)
    output_text = ""
    for segment in segments:
        # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        output_text += segment.text + ""

    return output_text


@app.post("/answer")
async def rag_api(question: str = Form(None), audio: UploadFile = File(None)):
    if not question and not audio:
        raise HTTPException(status_code=400, detail="Bạn phải cung cấp ít nhất 'question' hoặc 'audio'")
    
    if question:
        # xử lý câu hỏi văn bản
        return {"type": "text", "content": question}

    if audio:
        # xử lý âm thanh, ví dụ chuyển speech to text
        content = transcribe_audio(audio)
        if not content:
            raise HTTPException(status_code=400, detail="Can't transcribe audio")
        return {"type": "audio", "filename": audio.filename, "size": len(content)}

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