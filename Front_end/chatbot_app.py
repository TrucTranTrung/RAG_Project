import streamlit as st
import uuid
from datetime import datetime
import textwrap
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import io
import wave
import requests
import base64
import numpy as np

# --- URL BACKEND ---
CHATBOT_URL = "http://localhost:4096/answer"
WHISPER_URL = "http://localhost:8000/STT/"
TTS_URL = "http://localhost:8001/transcribe"
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# --- Hàm gọi backend ---
def send_to_chatbot(prompt):
    try:
        res = requests.post(
            CHATBOT_URL,
            files={"question": (None, prompt)}
        )
        if res.status_code == 200:
            return res.json().get("content", "Không có phản hồi.")
        else:
            return f"Lỗi chatbot: {res.status_code} | {res.text}"
    except Exception as e:
        return f"Lỗi gọi chatbot: {e}"

def speech_to_text(audio_bytes):
    try:
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        res = requests.post(WHISPER_URL, files=files)
        if res.status_code == 200:
            return res.json().get("output_text", "")
        return ""
    except Exception:
        return ""

# --- Setup Page ---
st.set_page_config(page_title="Chatbot", layout="wide")

# --- Initialize Session State ---
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "show_recorder" not in st.session_state:
    st.session_state.show_recorder = False

# --- Sidebar ---
with st.sidebar:
    st.title("🗂️ Lịch sử Chat")
    if st.button("➕ Cuộc trò chuyện mới"):
        new_chat_id = str(uuid.uuid4())
        st.session_state.chats[new_chat_id] = {
            "title": "Cuộc trò chuyện mới",
            "messages": [],
            "created_at": datetime.now().strftime("%H:%M %d-%m-%Y")
        }
        st.session_state.current_chat = new_chat_id

    st.markdown("---")

    for chat_id, chat_data in st.session_state.chats.items():
        label = chat_data["title"]
        if st.button(label, key=chat_id):
            st.session_state.current_chat = chat_id

# --- Main Chat Interface ---
st.markdown(
    """
    <style>
    div.stChatInputContainer {position: fixed; bottom: 0; left: 0; right: 0; background: white; padding: 10px;}
    div.block-container {padding-bottom: 70px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💬 Chatbot")

if st.session_state.current_chat is None:
    st.info("Hãy nhấn 'Cuộc trò chuyện mới' để bắt đầu.")
else:
    chat = st.session_state.chats[st.session_state.current_chat]

    # Hiển thị các tin nhắn đã lưu
    for msg in chat["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Thanh nhập câu hỏi ---
    col1, col2 = st.columns([10, 1])
    with col1:
        prompt = st.chat_input("Nhập câu hỏi...")
    with col2:
        if st.button("🎙️", key="mic_toggle"):
            st.session_state.show_recorder = not st.session_state.show_recorder

    # --- Xử lý gửi ---
    if prompt:
        # Thêm tin nhắn người dùng
        chat["messages"].append({"role": "user", "content": prompt})

        # Gọi bot
        response = send_to_chatbot(prompt)
        chat["messages"].append({"role": "assistant", "content": response})

        # Cập nhật tiêu đề
        if len(chat["messages"]) == 2:
            first_line = prompt.strip().split("\n")[0]
            short_title = textwrap.shorten(first_line, width=40, placeholder="...")
            chat["title"] = short_title or "Cuộc trò chuyện"

        # Reload để hiển thị ngay
        st.experimental_rerun()

    # --- Ghi âm ---
    if st.session_state.show_recorder:
        st.markdown("🎙️ **Đang ghi âm...**")

        class AudioProcessor(AudioProcessorBase):
            def __init__(self):
                self.frames = []
                self.volume_level = 0

            def recv_audio(self, frame):
                self.frames.append(frame)
                audio = frame.to_ndarray()
                self.volume_level = np.sqrt(np.mean(audio ** 2))
                return frame

        ctx = webrtc_streamer(
            key="audio",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"audio": True, "video": False},
            audio_receiver_size=1024,
            audio_processor_factory=AudioProcessor,
        )

        if ctx.audio_processor:
            st.write(f"Mức âm lượng: {ctx.audio_processor.volume_level:.4f}")

            if st.button("Dừng & Gửi"):
                frames = ctx.audio_processor.frames
                if frames:
                    pcm_data = b""
                    for f in frames:
                        pcm_data += f.to_ndarray().tobytes()

                    wav_buffer = io.BytesIO()
                    with wave.open(wav_buffer, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(48000)
                        wf.writeframes(pcm_data)

                    audio_data = wav_buffer.getvalue()
                    text_transcribed = speech_to_text(audio_data)

                    # Thêm tin nhắn từ mic
                    chat["messages"].append({"role": "user", "content": text_transcribed})
                    response = send_to_chatbot(text_transcribed)
                    chat["messages"].append({"role": "assistant", "content": response})
                    st.rerun()

