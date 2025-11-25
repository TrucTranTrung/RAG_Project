import os
import time
import requests
import pytest

# ================= CONFIG =================
SERVER_URL = os.environ.get("SERVER_URL", "http://tts-api:8001")
TRANSCRIBE_PATH = "/transcribe"

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

def post_with_retry(url, data=None, files=None, retries=5, delay=1):
    """Gửi request POST với retry nếu connection error"""
    for i in range(retries):
        try:
            response = requests.post(url, data=data, files=files, timeout=10)
            return response
        except requests.exceptions.ConnectionError:
            if i < retries - 1:
                time.sleep(delay)
            else:
                raise

# Fixture đảm bảo server sẵn sàng
@pytest.fixture(scope="module", autouse=True)
def ensure_server_ready():
    assert wait_for_server(SERVER_URL), f"Server {SERVER_URL} không reachable"

# ================= TEST CASES =================
def test_transcribe_audio_success():
    """Test thành công với text hợp lệ"""
    data = {"text_input": "Hello world"}
    response = post_with_retry(SERVER_URL + TRANSCRIBE_PATH, data=data)
    assert response.status_code == 200
    json_data = response.json()
    assert "audio_base64" in json_data
    assert isinstance(json_data["audio_base64"], str)
    assert len(json_data["audio_base64"]) > 0

def test_transcribe_missing_input():
    """Test lỗi khi text_input thiếu"""
    response = post_with_retry(SERVER_URL + TRANSCRIBE_PATH, data={})
    assert response.status_code == 422

def test_transcribe_invalid_type():
    """Test input không hợp lệ (ví dụ gửi file text)"""
    files = {"file": ("test.txt", b"not audio")}
    response = post_with_retry(SERVER_URL + TRANSCRIBE_PATH, files=files)
    # Tùy cách API xử lý input invalid, 400 hoặc 422
    assert response.status_code in (400, 422)
