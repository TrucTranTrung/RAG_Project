# test_chatbot_api_full.py
import io
import base64
import pytest
import requests
from unittest.mock import patch, MagicMock

SERVER_URL = "http://chatbot_api/answer"
TIMEOUT = 5


def _post_form(data=None, files=None):
    """Gửi POST tới API với timeout, fail test nếu không connect được"""
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
@patch("requests.post")
def test_audio_only_returns_base64(mock_post):
    """
    Gửi audio-only, mock STT trả text giả.
    Kiểm tra API trả audio_base64 decode được.
    """

    # --- Mock STT response ---
    mock_stt = MagicMock()
    mock_stt.status_code = 200
    mock_stt.json.return_value = {"output_text": "transcribed audio"}
    mock_stt.raise_for_status = lambda: None
    mock_post.return_value = mock_stt

    # --- Fake audio file ---
    fake_audio = io.BytesIO(b"FAKE_WAV_BYTES")
    files = {"audio": ("test.wav", fake_audio, "audio/wav")}

    resp = _post_form(files=files)
    assert resp.status_code == 200

    data = resp.json()
    assert "audio_base64" in data

    # Kiểm tra base64 decode được
    decoded = base64.b64decode(data["audio_base64"])
    assert isinstance(decoded, bytes)
    assert len(decoded) > 0


# ==========================
# Missing input test
# ==========================
def test_no_input_returns_4xx():
    resp = _post_form(data={})
    # Status code phải là 4xx
    assert 400 <= resp.status_code < 500
    # Nếu trả JSON, ít nhất có key 'error'
    try:
        j = resp.json()
        assert "error" in j or j.get("detail") is not None
    except ValueError:
        # không trả JSON vẫn OK
        pass
