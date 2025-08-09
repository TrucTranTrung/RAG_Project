import streamlit as st
import uuid
from datetime import datetime
import textwrap
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import io
import wave
import requests
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

   # --- Ghi âm ---
if st.session_state.get("show_recorder", False):
    st.markdown("🎙️ **Đang ghi âm...**")

    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.frames = []
            self.volume_level = 0.0

        def recv(self, frame: av.AudioFrame):
            audio = frame.to_ndarray()
            self.frames.append(audio)
            self.volume_level = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
            return frame

    ctx = webrtc_streamer(
        key="audio_recorder",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"audio": True, "video": False},
        audio_receiver_size=1024,
        audio_processor_factory=AudioProcessor,
    )

    if ctx and ctx.audio_processor:
        st.write(f"Mức âm lượng: {ctx.audio_processor.volume_level:.4f}")

        if st.button("🛑 Dừng & Nhận diện 🎤", key="stop_send_mic_button"):
            frames = ctx.audio_processor.frames
            if not frames:
                st.warning("Không có dữ liệu âm thanh để gửi.")
            else:
                try:
                    # Gộp khung âm thanh
                    pcm_data = np.concatenate(frames, axis=0)

                    # Tạo file WAV
                    wav_buffer = io.BytesIO()
                    with wave.open(wav_buffer, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit = 2 bytes
                        wf.setframerate(48000)
                        wf.writeframes(pcm_data.astype(np.int16).tobytes())
                    audio_data = wav_buffer.getvalue()

                    # Gửi tới API STT
                    with st.spinner("🎧 Đang chuyển giọng nói thành văn bản..."):
                        text_transcribed = speech_to_text(audio_data)

                    if text_transcribed:
                        st.session_state.transcribed_audio = audio_data
                        st.session_state.transcribed_text = text_transcribed
                        st.session_state.waiting_to_send = True
                    else:
                        st.warning("❗ Không thể chuyển đổi giọng nói thành văn bản.")
                except Exception as e:
                    st.error(f"⚠️ Lỗi khi xử lý âm thanh: {e}")

                st.session_state.show_recorder = False
                st.rerun()

# --- Gửi hoặc Ghi lại ---
if st.session_state.get("waiting_to_send", False):
    st.markdown("📝 **Bạn đã nói:**")
    st.text_area("Nội dung", st.session_state.transcribed_text, height=100)

    # 🔊 Phát lại âm thanh đã ghi
    if st.session_state.get("transcribed_audio"):
        st.audio(st.session_state.transcribed_audio, format="audio/wav")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Gửi vào chatbot"):
            prompt = st.session_state.transcribed_text
            with st.spinner("🤖 Đang gửi đến chatbot..."):
                response = send_to_chatbot(prompt)

            chat["messages"].append({"role": "user", "content": prompt})
            chat["messages"].append({"role": "assistant", "content": response})

            if len(chat["messages"]) == 2:
                first_line = prompt.strip().split("\n")[0]
                short_title = textwrap.shorten(first_line, width=40, placeholder="...")
                chat["title"] = short_title or "Cuộc trò chuyện"

            # Reset trạng thái
            st.session_state.waiting_to_send = False
            st.session_state.transcribed_audio = None
            st.session_state.transcribed_text = ""
            st.rerun()

    with col2:
        if st.button("🔁 Ghi lại"):
            st.session_state.waiting_to_send = False
            st.session_state.transcribed_audio = None
            st.session_state.transcribed_text = ""
            st.session_state.show_recorder = True
            st.rerun()