# test_chatbot_api.py

import pytest
import wave
import os
import sys
import requests  # Thêm import này để sửa lỗi NameError
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# --- Cấu hình ---
# Tên file API chính của bạn, không có đuôi .py
MAIN_API_MODULE = "API_LLAMA3_2"  


# --- Fixture để thiết lập môi trường và TestClient ---
@pytest.fixture(scope="module")
def client():
    """
    Fixture này thực hiện 3 việc quan trọng:
    1. Thiết lập môi trường: Thay đổi thư mục làm việc và thêm đường dẫn
       vào sys.path để code API có thể tìm thấy các file phụ trợ.
    2. Mock các hàm gây lỗi khi import: Chặn get_pgvector_store ngay từ đầu.
    3. Tạo ra một TestClient: Cung cấp một client để gọi API mà không cần
       chạy server thực sự, cho phép các mock khác hoạt động.
    """
    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)

    test_file_path = Path(__file__).resolve()
    project_root = test_file_path.parent.parent
    api_folder_path = project_root / "src" / "services" / "chatbot_api"

    
    if not api_folder_path.is_dir():
         raise FileNotFoundError(f"Không tìm thấy thư mục API tại: {api_folder_path}")

    os.chdir(api_folder_path)
    sys.path.insert(0, str(api_folder_path))

    # Mock hàm gây lỗi TRƯỚC KHI import app
    with patch('db.get_pgvector_store', return_value=MagicMock()) as mock_store:
        # Import app SAU KHI đã thiết lập môi trường và mock cần thiết
        from API_LLAMA3_2 import app

        
        # `yield` trả về TestClient cho các hàm test sử dụng
        yield TestClient(app)

    # Dọn dẹp sau khi test xong
    os.chdir(original_cwd)
    sys.path = original_sys_path


# --- Fixture để tạo và dọn dẹp file âm thanh giả ---
@pytest.fixture
def fake_wav_file():
    """
    Fixture này tạo ra một file WAV im lặng tạm thời cho việc test.
    """
    fake_filename = "temp_rag_audio.wav"
    with wave.open(fake_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b'\x00' * 16000)

    yield fake_filename
    os.remove(fake_filename)


# --- Test Case 1: Luồng văn bản thành công ---
@patch(f'{MAIN_API_MODULE}.query_similar_vectors_from_pgvector')
@patch(f'{MAIN_API_MODULE}.get_top_k_contexts')
@patch(f'{MAIN_API_MODULE}.get_entities_as_string_GEMINI')
def test_rag_api_with_question(mock_gemini, mock_rerank, mock_query_pgvector, client):
    """
    Kiểm tra luồng xử lý khi nhận đầu vào là một câu hỏi dạng text.
    """
    # Sửa lỗi AttributeError: Mock phải trả về một đối tượng có thuộc tính .page_content
    mock_doc = MagicMock()
    mock_doc.page_content = "This is a mock document content."
    mock_query_pgvector.return_value = [(mock_doc, 0.9)]
    
    mock_rerank.return_value = ["reranked_doc1"]
    mock_gemini.return_value = "This is the final answer from Gemini."
    
    question_text = "What is java?"
    response = client.post("http://127.0.0.1:4096/answer", data={"question": question_text})

    assert response.status_code == 200
    expected_response = {
        "type": "text",
        "content": "This is the final answer from Gemini."
    }
    assert response.json() == expected_response
    mock_gemini.assert_called_once()


# --- Test Case 2: Luồng âm thanh thành công ---
@patch(f'{MAIN_API_MODULE}.requests.post')
@patch(f'{MAIN_API_MODULE}.query_similar_vectors_from_pgvector')
@patch(f'{MAIN_API_MODULE}.get_top_k_contexts')
@patch(f'{MAIN_API_MODULE}.get_entities_as_string_GEMINI')
def test_rag_api_with_audio(mock_gemini, mock_rerank, mock_query_pgvector, mock_requests_post, client, fake_wav_file):
    """
    Kiểm tra luồng xử lý khi nhận đầu vào là một file âm thanh.
    """
    # Sửa lỗi AttributeError: Mock phải trả về một đối tượng có thuộc tính .page_content
    mock_doc = MagicMock()
    mock_doc.page_content = "This is a mock document content for audio."
    mock_query_pgvector.return_value = [(mock_doc, 0.95)]

    mock_rerank.return_value = ["reranked_doc_audio"]
    mock_gemini.return_value = "This is the audio answer from Gemini."

    def mock_api_calls(url, **kwargs):
        mock_response = MagicMock()
        if "whisper-api" in url:
            mock_response.status_code = 200
            mock_response.json.return_value = {"output_text": "Transcribed audio text"}
        elif "tts-api" in url:
            mock_response.status_code = 200
            mock_response.json.return_value = {"output_sound": "fake_base64_audio_string"}
        else:
            mock_response.status_code = 404
        return mock_response
    
    mock_requests_post.side_effect = mock_api_calls

    with open(fake_wav_file, 'rb') as f:
        files = {'audio': (os.path.basename(fake_wav_file), f, 'audio/wav')}
        response = client.post("http://127.0.0.1:4096/answer", files=files)

    assert response.status_code == 200
    expected_response = {
        "type": "audio",
        "audio_base64": "fake_base64_audio_string"
    }
    assert response.json() == expected_response
    assert mock_requests_post.call_count == 2


# --- Test Case 3: Lỗi không có input ---
def test_rag_api_missing_input(client):
    """
    Kiểm tra API trả về lỗi 400 khi không có câu hỏi hoặc file âm thanh.
    """
    response = client.post("/answer")
    assert response.status_code == 400
    assert response.json() == {"error": "No question or audio provided"}


# --- Test Case 4: Lỗi khi service STT thất bại ---
@patch(f'{MAIN_API_MODULE}.logger.error')  
@patch(f'{MAIN_API_MODULE}.requests.post')
def test_rag_api_stt_service_fails(mock_requests_post, mock_logger, client, fake_wav_file):
    """
    Kiểm tra API trả về lỗi 500 khi không thể kết nối đến STT service.
    Đồng thời, mock logger để không in ra lỗi không mong muốn trong lúc test.
    """
    # Lỗi NameError được sửa bằng cách thêm `import requests` ở đầu file
    mock_requests_post.side_effect = requests.exceptions.RequestException("STT service is down")

    with open(fake_wav_file, 'rb') as f:
        files = {'audio': (os.path.basename(fake_wav_file), f, 'audio/wav')}
        response = client.post("http://127.0.0.1:4096/answer", files=files)

    # Kiểm tra kết quả
    assert response.status_code == 500
    assert "error" in response.json()
    
    # Kiểm tra xem logger có được gọi đúng cách không
    mock_logger.assert_called_once()
