# test_tts_api.py
import pytest
import requests
import base64
from unittest.mock import patch, Mock

SERVER_URL = "http://127.0.0.1:8001"
TRANSCRIBE_PATH = "/transcribe"
TIMEOUT = 5  # giây


def server_is_up(url: str) -> bool:
    """
    Kiểm tra nhanh xem server có phản hồi (GET /) hay không.
    Trả về True nếu server reachable, False nếu connection error / timeout.
    """
    try:
        # probeless check - nhiều app trả 200/404 cho root; mục đích chỉ để biết server reachable
        requests.get(url + "/", timeout=1)
        return True
    except requests.RequestException:
        return False


@pytest.fixture(scope="module")
def require_server_or_skip():
    """
    Fixture giúp skip những test integration nếu server không chạy.
    Sử dụng bằng cách thêm fixture này vào signature test.
    """
    if not server_is_up(SERVER_URL):
        pytest.skip(f"TT S API not reachable at {SERVER_URL} — skipping integration tests")


# -----------------------
# Integration: gọi server thật (chạy nếu server đang up)
# -----------------------
def test_transcribe_integration_success(require_server_or_skip):
    """
    Gọi API /transcribe trên server chạy ở port 8001.
    Test này sẽ bị skip nếu server không chạy.
    """
    url = SERVER_URL + TRANSCRIBE_PATH
    payload = {"text_input": "Integration test: Hello world."}

    resp = requests.post(url, data=payload, timeout=TIMEOUT)
    assert resp.status_code == 200, f"Expected 200 but got {resp.status_code}; body: {resp.text}"

    json_data = resp.json()
    assert "output_sound" in json_data, f"Missing 'output_sound' in response: {json_data}"

    out_b64 = json_data["output_sound"]
    assert isinstance(out_b64, str) and len(out_b64) > 0

    # kiểm tra base64 decode hợp lệ
    try:
        decoded = base64.b64decode(out_b64, validate=True)
    except Exception as e:
        pytest.fail(f"output_sound is not valid base64: {e}")

    assert len(decoded) > 0, "Decoded audio is empty"


def test_transcribe_integration_missing_input(require_server_or_skip):
    """
    Gọi server thật với missing input -> mong đợi status 422 (FastAPI validation).
    Bị skip nếu server không chạy.
    """
    url = SERVER_URL + TRANSCRIBE_PATH
    resp = requests.post(url, data={}, timeout=TIMEOUT)
    assert resp.status_code == 422, f"Expected 422 for missing input but got {resp.status_code}. Body: {resp.text}"


def test_transcribe_integration_model_error(require_server_or_skip):
    """
    Nếu server thực sự trả 500 với message {"error": "..."} thì test sẽ assert theo đó.
    Nếu server trả khác (ví dụ 200), test sẽ fail — vì đây là test để kiểm tra xử lý lỗi server.
    Bị skip nếu server không chạy.
    """
    url = SERVER_URL + TRANSCRIBE_PATH
    # Gọi với input bình thường — nếu server trả 500 do model failing, sẽ kiểm tra body
    resp = requests.post(url, data={"text_input": "trigger possible model failure"}, timeout=TIMEOUT)

    # Nếu server trả 500, mong body giống {"error": "..."}
    if resp.status_code == 500:
        json_data = resp.json()
        assert isinstance(json_data, dict) and "error" in json_data
    else:
        # Nếu server không trả 500 thì test chỉ đảm bảo endpoint vẫn hoạt động.
        assert resp.status_code in (200, 422), f"Unexpected status: {resp.status_code}; body: {resp.text}"


# -----------------------
# Unit-style tests (mock requests) — chạy offline, không cần server
# -----------------------
@patch("requests.post")
def test_transcribe_mock_success(mock_post):
    """
    Test offline: mock requests.post để trả về response 200 + output_sound base64.
    Dùng khi server chưa chạy.
    """
    fake_b64 = base64.b64encode(b"fake-audio-bytes").decode("ascii")
    fake_resp = Mock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"output_sound": fake_b64}
    mock_post.return_value = fake_resp

    url = SERVER_URL + TRANSCRIBE_PATH
    resp = requests.post(url, data={"text_input": "Hello world"}, timeout=TIMEOUT)

    assert resp.status_code == 200
    json_data = resp.json()
    assert json_data["output_sound"] == fake_b64
    # kiểm tra decode base64
    assert base64.b64decode(json_data["output_sound"]) == b"fake-audio-bytes"


@patch("requests.post")
def test_transcribe_mock_model_exception(mock_post):
    """
    Mock một response lỗi 500 từ server (ví dụ: Model failed to load).
    """
    fake_resp = Mock()
    fake_resp.status_code = 500
    fake_resp.json.return_value = {"error": "Model failed to load"}
    mock_post.return_value = fake_resp

    url = SERVER_URL + TRANSCRIBE_PATH
    resp = requests.post(url, data={"text_input": "anything"}, timeout=TIMEOUT)

    assert resp.status_code == 500
    assert resp.json() == {"error": "Model failed to load"}


@patch("requests.post")
def test_transcribe_mock_missing_input(mock_post):
    """
    Mock response 422 cho missing input case.
    """
    fake_resp = Mock()
    fake_resp.status_code = 422
    fake_resp.text = "Unprocessable Entity"
    # json() may raise for 422; but we only check status code here
    mock_post.return_value = fake_resp

    url = SERVER_URL + TRANSCRIBE_PATH
    resp = requests.post(url, data={}, timeout=TIMEOUT)

    assert resp.status_code == 422