import streamlit as st 
import uuid
from datetime import datetime
import textwrap
from st_audiorec import st_audiorec

# --- Setup Page ---
st.set_page_config(page_title="Chatbot", layout="wide")

# --- Initialize Session State ---
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "show_recorder" not in st.session_state:
    st.session_state.show_recorder = False
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

# --- Sidebar ---
with st.sidebar:
    st.title("📚 Lịch sử Chat")

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

        if st.button(label="", key=chat_id):
            st.session_state.current_chat = chat_id

        st.markdown(
            f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 0.25rem;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                font-size: 15px;
                padding: 0.25rem 0.5rem;
                border: 1px solid #DDD;
                border-radius: 6px;
                margin-bottom: 5px;
                background-color: #f9f9f9;
            " title="{label}">
                📁 <span>{label}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# --- Main Chat Interface ---
st.markdown("<h1 style='text-align: center;'>🤖 Chatbot</h1>", unsafe_allow_html=True)

if st.session_state.current_chat is None:
    st.info("Hãy nhấn '➕ Cuộc trò chuyện mới' để bắt đầu.")
else:
    chat = st.session_state.chats[st.session_state.current_chat]

    # Hiển thị các tin nhắn đã gửi
    for msg in chat["messages"]:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "audio":
                st.audio(msg["content"], format="audio/wav")
            else:
                st.markdown(msg["content"])

    # --- Input + Mic Button ---
    col1, col2 = st.columns([8, 1])
    with col1:
        prompt = st.chat_input("Nhập câu hỏi...")
    with col2:
        if st.button("🎤", use_container_width=True):
            st.session_state.show_recorder = not st.session_state.show_recorder

    # --- Xử lý nhập văn bản ---
    if prompt:
        chat["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if len(chat["messages"]) == 1:
            first_line = prompt.strip().split("\n")[0]
            short_title = textwrap.shorten(first_line, width=40, placeholder="...")
            chat["title"] = short_title or "Cuộc trò chuyện"

        response = f"🤖 Tôi đã nhận được: **{prompt}**"
        chat["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    # --- Giao diện ghi âm nhỏ gọn ---
    if st.session_state.show_recorder:
        with st.container():
            st.markdown("### 🎙 Ghi âm")
            

            audio_data = st_audiorec()

            if audio_data:
                st.session_state.last_audio = audio_data
                st.audio(audio_data, format="audio/wav")

                if st.button("✅ Gửi"):
                    chat["messages"].append({
                        "role": "user",
                        "content": audio_data,
                        "type": "audio"
                    })
                    with st.chat_message("user"):
                        st.audio(audio_data, format="audio/wav")

                    response = "🤖 Tôi đã nhận được đoạn ghi âm!"
                    chat["messages"].append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.markdown(response)

                    st.session_state.last_audio = None
                    st.session_state.show_recorder = False