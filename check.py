import streamlit as st
from st_audiorec import st_audiorec

st.title("🎙️ Demo Ghi âm & Nhận diện")

# Recorder phải luôn được render
audio_data = st_audiorec()

if st.session_state.get("show_recorder", False):
    st.markdown("🎙️ **Đang ghi âm...**")

    if st.button("🛑 Dừng & Nhận diện 🎤", key="stop_send_mic_button"):
        try:
            if audio_data:
                with open("test.wav", "wb") as f:
                    f.write(audio_data)

                with st.spinner("🎧 Đang chuyển giọng nói thành văn bản..."):
                    text_transcribed = speech_to_text(audio_data)

                if text_transcribed:
                    st.session_state.transcribed_audio = audio_data
                    st.session_state.transcribed_text = text_transcribed
                    st.session_state.waiting_to_send = True
                else:
                    st.warning("❗ Không thể chuyển đổi giọng nói thành văn bản.")
            else:
                st.error("⚠️ Chưa có dữ liệu ghi âm!")
        except Exception as e:
            st.error(f"⚠️ Lỗi khi xử lý âm thanh: {e}")

        st.session_state.show_recorder = False
        st.rerun()