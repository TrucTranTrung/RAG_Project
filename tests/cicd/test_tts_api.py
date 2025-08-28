# test_tts_api.py

import requests
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np
import base64
import sys
import os
from pathlib import Path

# --- Fixture để thiết lập môi trường và TestClient ---
@pytest.fixture(scope="module")
def app_client():
    """
    Fixture này thực hiện 3 việc quan trọng:
    1. Thiết lập môi trường: Thay đổi thư mục làm việc (chdir) và thêm đường dẫn
       vào sys.path để code API có thể tìm thấy các file model.
    2. Tạo ra một TestClient: Cung cấp một client để gọi API mà không cần
       chạy server thực sự.
    3. Dọn dẹp môi trường: Trả lại thư mục làm việc và sys.path về như cũ
       sau khi tất cả các test trong module đã chạy xong.
    """
    # Lưu lại trạng thái ban đầu
    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)

    # Xác định và chuyển đến thư mục làm việc của API
    # Path(__file__) -> thư mục chứa file test này
    # .parent.parent -> đi lùi ra 2 cấp để đến thư mục gốc của dự án
    # Nối với đường dẫn còn lại để vào thư mục Text_to_Speech
    api_folder_path = Path(__file__).parent.parent / "src" / "services" / "Text_to_Speech"
    
    if not api_folder_path.exists():
        raise FileNotFoundError(
            f"Thư mục API không tìm thấy tại: {api_folder_path}. "
            "Hãy đảm bảo cấu trúc thư mục của bạn là chính xác."
        )

    os.chdir(api_folder_path)
    sys.path.insert(0, str(api_folder_path))

    # Import `app` SAU KHI đã thay đổi thư mục
    from TTS_API import app

    # `yield` trả về TestClient cho các hàm test sử dụng
    yield TestClient(app)

    # --- Code dọn dẹp (chạy sau khi test xong) ---
    os.chdir(original_cwd)
    sys.path = original_sys_path


# --- Test Case 1: Trường hợp thành công ---
# Đường dẫn patch bây giờ là 'TTS_API.LFinference' vì chúng ta test module TTS_API
@patch('TTS_API.LFinference')
@patch('TTS_API.torch.randn')
def test_transcribe_success(mock_torch_randn, mock_LFinference, app_client):
    """
    Kiểm tra endpoint hoạt động đúng với input hợp lệ.
    - Mock model LFinference để trả về dữ liệu âm thanh giả.
    - Sử dụng TestClient để gọi API trực tiếp, nhanh hơn và không cần server.
    """
    # 1. --- Thiết lập Mock (Arrange) ---
    fake_audio_wave = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mock_LFinference.return_value = (fake_audio_wave, "fake_state")
    mock_torch_randn.return_value = "fake_tensor"
    input_text = "Random sentence for testing."

    # 2. --- Hành động (Act) ---
    response = requests.post("http://127.0.0.1:8001/transcribe/", data={"text_input": input_text})

    # 3. --- Kiểm tra (Assert) ---
    assert response.status_code == 200
    
    # Kiểm tra response có phải là JSON hợp lệ và có chứa key mong muốn không
    response_json = response.json()
    assert "output_sound" in response_json
    
    # Kiểm tra xem giá trị có phải là một chuỗi base64 không rỗng hay không
    output_sound = response_json["output_sound"]
    assert isinstance(output_sound, str)
    assert len(output_sound) > 0


# --- Test Case 2: Trường hợp Model gặp lỗi ---
@patch('TTS_API.LFinference', side_effect=Exception("Model failed to load"))
def test_transcribe_model_exception(mock_LFinference, app_client): # Thêm app_client
    """
    Kiểm tra API trả về lỗi 500 khi model LFinference gặp sự cố.
    """
    # Sử dụng app_client
    response = app_client.post("/transcribe", data={"text_input": "Random sentence."})

    assert response.status_code == 500
    assert response.json() == {"error": "Model failed to load"}


# --- Test Case 3: Trường hợp Input không hợp lệ ---
def test_transcribe_missing_input(app_client): # Thêm app_client
    """
    Kiểm tra API trả về lỗi 422 khi không có 'text_input' được cung cấp.
    """
    # Sử dụng app_client
    response = app_client.post("/transcribe", data={})

    assert response.status_code == 422
