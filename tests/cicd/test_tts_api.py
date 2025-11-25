# test_tts_api.py

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np
import sys
import os
from pathlib import Path

# --- Fixture để tạo TestClient ---
@pytest.fixture(scope="module")
def app_client():
    """
    Thiết lập môi trường và trả về TestClient.
    """
    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)

    # Thư mục chứa API
    api_folder = Path(__file__).parent.parent.parent / "src" / "services" / "Text_to_Speech"
    if not api_folder.exists():
        raise FileNotFoundError(f"Thư mục API không tồn tại: {api_folder}")

    os.chdir(api_folder)
    sys.path.insert(0, str(api_folder))

    # Import app SAU khi thay đổi đường dẫn
    from TTS_API import app
    client = TestClient(app)

    yield client

    # Dọn dẹp sau test
    os.chdir(original_cwd)
    sys.path = original_sys_path


# --- Test Case 1: Thành công với mock ---
@patch('TTS_API.LFinference')
@patch('TTS_API.StyleTTS2.models._load_model')  # mock load model để tránh cần file .pth
@patch('TTS_API.torch.randn')
def test_transcribe_success(mock_torch_randn, mock_load_model, mock_LFinference, app_client):
    # Mock load_model trả về object giả
    mock_load_model.return_value = "fake_model"

    # Mock LFinference trả về audio giả
    fake_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mock_LFinference.return_value = (fake_audio, "fake_state")
    
    # Mock torch.randn nếu TTS API dùng
    mock_torch_randn.return_value = "fake_tensor"

    # Gọi API
    response = app_client.post("/transcribe", data={"text_input": "Hello world"})

    assert response.status_code == 200
    json_data = response.json()
    assert "output_sound" in json_data
    assert isinstance(json_data["output_sound"], str)
    assert len(json_data["output_sound"]) > 0


# --- Test Case 2: Model gặp lỗi ---
@patch('TTS_API.LFinference', side_effect=Exception("Model failed to load"))
def test_transcribe_model_exception(mock_LFinference, app_client):
    response = app_client.post("/transcribe", data={"text_input": "Random sentence."})

    assert response.status_code == 500
    assert response.json() == {"error": "Model failed to load"}


# --- Test Case 3: Thiếu input ---
def test_transcribe_missing_input(app_client):
    response = app_client.post("/transcribe", data={})
    assert response.status_code == 422
