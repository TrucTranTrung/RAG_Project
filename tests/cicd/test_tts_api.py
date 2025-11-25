import os
import time
import requests
import pytest

# ================= CONFIG =================
SERVER_URL = os.environ.get("SERVER_URL", "http://tts-api:8001")
TRANSCRIBE_PATH = "/transcribe/"

# ================= HELPER =================
def wait_for_server(url, timeout=30):
    """Đợi server TTS sẵn sàng trước khi chạy test"""
    end = time.time() + timeout
    while time.time() < end:
        try:
            r = requests.get(url + "/docs", timeout=1)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            time.sleep(0.5)
    return False

# Fixture đảm bảo server sẵn sàng
@pytest.fixture(scope="module", autouse=True)
def ensure_server_ready():
    assert wait_for_server(SERVER_URL), f"Server {SERVER_URL} không reachable"

# ================= TEST CASES =================
def test_transcribe_audio_success():
    """Test thành công với text hợp lệ"""
    data = {"text_input": "Hello world"}
    response = requests.post(SERVER_URL + TRANSCRIBE_PATH, data=data)
    assert response.status_code == 200
    json_data = response.json()
    assert "audio_base64" in json_data
    assert isinstance(json_data["audio_base64"], str)
    assert len(json_data["audio_base64"]) > 0

def test_transcribe_missing_input():
    """Test lỗi khi text_input thiếu"""
    response = requests.post(SERVER_URL + TRANSCRIBE_PATH, data={})
    assert response.status_code == 422

def test_transcribe_invalid_type():
    """Test input không hợp lệ (ví dụ gửi file text)"""
    files = {"file": ("test.txt", b"not audio")}
    response = requests.post(SERVER_URL + TRANSCRIBE_PATH, files=files)
    assert response.status_code in (400, 422)
