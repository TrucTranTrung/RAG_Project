#!/bin/bash
cd Container_Folder
pytest ../tests/cicd/test_tts_api.py
pytest ../tests/cicd/test_whisper-api.py
pytest ../tests/cicd/test_chatbot_api.py
