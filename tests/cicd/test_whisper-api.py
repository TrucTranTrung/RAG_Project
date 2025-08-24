# test_stt_api.py

import requests
import pytest
import wave
import os

# --- Cấu hình ---
API_BASE_URL = "http://127.0.0.1:8000" 


# --- Fixture để tạo và dọn dẹp file âm thanh giả ---
@pytest.fixture
def fake_wav_file():
    """
    Fixture này tạo ra một file WAV im lặng tạm thời cho việc test.
    Nó sẽ tự động xóa file sau khi bài test sử dụng nó hoàn tất.
    """
    fake_filename = "temp_test_audio.wav"
    sample_rate = 16000  
    duration_seconds = 1
    n_frames = int(duration_seconds * sample_rate)
    
    # Tạo một file WAV im lặng
    with wave.open(fake_filename, 'wb') as wf:
        wf.setnchannels(1)  
        wf.setsampwidth(2)  
        wf.setframerate(sample_rate)
        wf.writeframes(b'\x00' * n_frames * 2) 

    # `yield` trả về đường dẫn file cho bài test
    yield fake_filename

    # Code dọn dẹp: Chạy sau khi bài test kết thúc
    os.remove(fake_filename)


@pytest.fixture
def invalid_audio_file():
    """
    Fixture này tạo ra một file text giả được đặt tên là .wav để kiểm tra xử lý lỗi.
    """
    invalid_filename = "invalid_audio.wav"
    with open(invalid_filename, "w") as f:
        f.write("this is not an audio file")
    
    yield invalid_filename
    
    os.remove(invalid_filename)


# --- Test Case 1: Trường hợp thành công ---
def test_transcribe_audio_success(fake_wav_file):
    """
    Kiểm tra endpoint /STT/ với một file WAV hợp lệ.
    """
    # 1. --- Chuẩn bị (Arrange) ---
    url = f"{API_BASE_URL}/STT/"
    file_path = fake_wav_file

    # 2. --- Hành động (Act) ---
    with open(file_path, 'rb') as f:
        # 'files' dictionary để gửi file dưới dạng multipart/form-data
        files = {'file': (os.path.basename(file_path), f, 'audio/wav')}
        response = requests.post(url, files=files)

    # 3. --- Kiểm tra (Assert) ---
    assert response.status_code == 200
    
    response_json = response.json()
    assert "language" in response_json
    assert "language_probability" in response_json
    assert "output_text" in response_json
    assert isinstance(response_json["output_text"], str)


# --- Test Case 2: Trường hợp file không hợp lệ ---
def test_transcribe_invalid_file_type(invalid_audio_file):
    """
    Kiểm tra API trả về lỗi 500 khi nhận một file không phải định dạng âm thanh.
    """
    url = f"{API_BASE_URL}/STT/"
    file_path = invalid_audio_file

    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, 'audio/wav')}
        response = requests.post(url, files=files)

    assert response.status_code == 500
    response_json = response.json()
    assert "error" in response_json


# --- Test Case 3: Trường hợp không gửi file ---
def test_transcribe_missing_file():
    """
    Kiểm tra API trả về lỗi 422 khi không có file nào được gửi.
    """
    url = f"{API_BASE_URL}/STT/"
    
    # Gửi request mà không có dictionary 'files'
    response = requests.post(url)

    assert response.status_code == 422
