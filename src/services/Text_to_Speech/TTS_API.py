import torch
import soundfile as sf
import numpy as np
import base64
import io
from models import LFinference
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
app = FastAPI()

@app.post("/transcribe")
async def transcribe_audio(text_input: str = Form(...)):
    try:
        sentences = text_input.split('.') # simple split by comma
        wavs = []
        s_prev = None
        for text in sentences:
            if text.strip() == "": continue
            text += '.' # add it back
            noise = torch.randn(1,1,256).to(device)
            wav, s_prev = LFinference(text, s_prev, noise, alpha=0.7, diffusion_steps=10, embedding_scale=1.5)
            wavs.append(wav)
        audio = np.concatenate(wavs)
        
         # Lưu vào buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio, samplerate=24000, format='WAV')
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        # end_time = time.time()
        # execution_time = end_time - start_time

        return JSONResponse(content={
            "output_sound": audio_base64
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})