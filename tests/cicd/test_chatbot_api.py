import os
import io
import wave
import base64
import pytest
import requests

SERVER_URL = os.environ.get("SERVER_URL", "http://chatbot_api:4096/answer")
TIMEOUT = 10  # tăng timeout cho audio

# ---------------------------
# Helper tạo WAV fake
# ---------------------------
def create_fake_wav_bytes(duration_sec=1, sample_rate=16000):
    buf = io.BytesIO()
    n_channels = 1
    sampwidth = 2  # 16-bit
    n_frames = duration_sec * sample_rate
    comptype = "NONE"
    compname = "not compressed"
    with wave.open(buf, 'wb') as wf:
        wf.setparams((n_channels, sampwidth, sample_rate, n_frames, comptype, compname))
        wf.writeframes(b'\x00\x00' * n_frames)
    buf.seek(0)
    return buf


def _post_form(data=None, files=None):
    try:
        resp = requests.post(SERVER_URL, data=data or {}, files=files or {}, timeout=TIMEOUT)
    except Exception as e:
        pytest.fail(f"Không thể kết nối tới chatbot API ({SERVER_URL}): {e}")
    return resp


# ==========================
# Text-only test
# ==========================
def test_text_only_returns_text():
    payload = {"question": "What is AI?"}
    resp = _post_form(data=payload)
    assert resp.status_code == 200
    j = resp.json()
    assert j.get("type") == "text"
    assert isinstance(j.get("content"), str)
    assert len(j.get("content").strip()) > 0


# ==========================
# Audio-only test
# ==========================
def test_audio_only_returns_base64():
    fake_audio_file = create_fake_wav_bytes()
    files = {"audio": ("fake.wav", fake_audio_file, "audio/wav")}
    
    resp = _post_form(files=files)
    assert resp.status_code == 200

    data = resp.json()
    assert "audio_base64" in data

    decoded = base64.b64decode(data["audio_base64"])
    assert isinstance(decoded, bytes)
    assert len(decoded) > 0


# ==========================
# Missing input test
# ==========================
def test_no_input_returns_4xx():
    resp = _post_form(data={})
    assert 400 <= resp.status_code < 500
    try:
        j = resp.json()
        assert "error" in j or j.get("detail") is not None
    except ValueError:
        pass
