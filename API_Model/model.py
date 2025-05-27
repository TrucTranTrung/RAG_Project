# --- Speech to text ---
from faster_whisper import WhisperModel

model_size = "medium"
model_whisper = WhisperModel(model_size, device="cpu", compute_type="int8")
