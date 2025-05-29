from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import tempfile
import shutil
import os

app = FastAPI()

# Load model only once at startup
model_size = "medium"
# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")


@app.post("/STT/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # start_time = time.time()
        suffix = os.path.splitext(file.filename)[-1]  # .wav, .mp3, .mp4...
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            temp_path = tmp_file.name
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
                
        # Transcribe
        segments, info = model.transcribe(temp_path, beam_size=5)
        # Ghép các đoạn văn bản lại
        output_text = ""
        for segment in segments:
            output_text += segment.text + " "

        # end_time = time.time()
        # execution_time = end_time - start_time

        return JSONResponse(content={
            "language": info.language,
            "language_probability": info.language_probability,
            "output_text": output_text.strip()
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
