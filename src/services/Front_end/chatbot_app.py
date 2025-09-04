import streamlit as st
import uuid
from datetime import datetime
import textwrap
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import io
import requests
import numpy as np
from st_audiorec import st_audiorec

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
    
def speech_to_text(audio_bytes: bytes):
    if not audio_bytes:
        return ""

    files = {"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")}
    response = requests.post(WHISPER_URL, files=files)

    if response.status_code == 200:
        print("STT status:", response.status_code)
        print("STT raw:", response.text)
        return response.json().get("output_text", "")
    else:
        print("STT error:", response.status_code, response.text)
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
        st.rerun()

    st.markdown("---")

    sorted_chats = sorted(
        st.session_state.chats.items(),
        key=lambda item: datetime.strptime(item[1]["created_at"], "%H:%M %d-%m-%Y"),
        reverse=True
    )

    for chat_id, chat_data in sorted_chats:
        label = chat_data["title"]
        is_current = chat_id == st.session_state.current_chat
        if st.button(label, key=chat_id):
            st.session_state.current_chat = chat_id
            st.rerun()

# --- Custom CSS ---
st.markdown(
    """
    <style>
    div.stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 10px;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
        z-index: 1000;
    }
    div.block-container {
        padding-bottom: 100px;
        padding-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💬 Chatbot")

if st.session_state.current_chat is None:
    st.info("Hãy nhấn '➕ Cuộc trò chuyện mới' để bắt đầu.")
else:
    chat = st.session_state.chats[st.session_state.current_chat]

    with st.container():
        for msg in chat["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    with st.container():
        col1, col2 = st.columns([10, 1])
        with col1:
            prompt = st.chat_input("Nhập câu hỏi...", key="chat_input_main")
        with col2:
            if st.button("🎙️", key="mic_toggle_button"):
                st.session_state.show_recorder = not st.session_state.show_recorder
                st.rerun()

    # --- Gửi tin nhắn văn bản ---
    if prompt:
        chat["messages"].append({"role": "user", "content": prompt})
        with st.spinner("Đang xử lý..."):
            response = send_to_chatbot(prompt)
        chat["messages"].append({"role": "assistant", "content": response})

        if len(chat["messages"]) == 2:
            first_line = prompt.strip().split("\n")[0]
            short_title = textwrap.shorten(first_line, width=40, placeholder="...")
            chat["title"] = short_title or "Cuộc trò chuyện"

        st.rerun()


# --- Audio Recorder ---
if st.session_state.get("show_recorder", False):
    audio_data = st_audiorec()
    if audio_data:
        # Dòng st.audio đã được vô hiệu hóa để không hiển thị trình phát âm thanh nữa
        # st.audio(audio_data, format="audio/wav")
        
        # Gán dữ liệu audio vào session state (nếu bạn cần dùng ở nơi khác)
        st.session_state.transcribed_audio = audio_data
        
        text_transcribed = speech_to_text(audio_data)

        if text_transcribed:
            st.session_state.transcribed_text = text_transcribed
            st.session_state.waiting_to_send = True
            prompt = st.session_state.transcribed_text
            
            with st.spinner("🤖 Đang gửi đến chatbot..."):
                response = send_to_chatbot(prompt)
            
            chat["messages"].append({"role": "user", "content": prompt})
            chat["messages"].append({"role": "assistant", "content": response})

            if len(chat["messages"]) == 2:
                first_line = prompt.strip().split("\n")[0]
                short_title = textwrap.shorten(first_line, width=40, placeholder="...")
                chat["title"] = short_title or "Cuộc trò chuyện"

            # Reset các trạng thái sau khi gửi
            st.session_state.waiting_to_send = False
            st.session_state.transcribed_audio = None
            st.session_state.transcribed_text = ""
            st.session_state.show_recorder = False
            st.session_state.already_sent = True   

            st.rerun()
        else:
            st.warning("❗ Không thể chuyển đổi giọng nói thành văn bản.")


