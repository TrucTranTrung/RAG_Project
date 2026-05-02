import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHATBOT_API_DIR = os.path.join(PROJECT_ROOT, "src", "services", "chatbot_api")
sys.path.insert(0, CHATBOT_API_DIR)

import model  # noqa: E402
from model import get_answer_from_context_GPT, get_answer_from_context_LOCAL  # noqa: E402


def test_local_answer_synthesizes_comparison_instead_of_dumping_context():
    context = [
        """
        ::clinical psychology a branch of psychology that studies, assesses,
        and treats people with psychological disorders.
        ::counseling psychology a branch of psychology that assists people with
        problems in living, often related to school, work, or marriage.
        """
    ]

    answer = get_answer_from_context_LOCAL(
        "",
        context,
        "What is the difference between clinical psychology and counseling psychology?",
    )

    assert "Demo local answer" not in answer
    assert "clinical psychology" in answer.lower()
    assert "counseling psychology" in answer.lower()
    assert "psychological disorders" in answer.lower()
    assert "problems in living" in answer.lower()


def test_local_answer_returns_not_enough_information_without_relevant_context():
    answer = get_answer_from_context_LOCAL(
        "",
        ["basic research pure science that aims to increase the scientific knowledge base."],
        "What is the difference between clinical psychology and counseling psychology?",
    )

    assert answer == "Not enough information to answer the question."


def test_local_provider_calls_ollama_chat_api(monkeypatch):
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": "Clinical psychology focuses on disorders."}}

    def fake_post(url, json, timeout):
        calls["url"] = url
        calls["json"] = json
        calls["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("CHAT_MODEL_PROVIDER", "local")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    monkeypatch.setattr(model.requests, "post", fake_post)

    answer = get_answer_from_context_GPT(
        "Context: {information}\nQuestion: {question}",
        ["clinical psychology treats psychological disorders."],
        "What is clinical psychology?",
    )

    assert answer == "Clinical psychology focuses on disorders."
    assert calls["url"] == "http://ollama.test:11434/api/chat"
    assert calls["json"]["model"] == "llama3.2"
    assert calls["json"]["stream"] is False


def test_gemini_provider_uses_api_gpt_key(monkeypatch):
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Gemini answer from RAG context."},
                            ],
                        },
                    },
                ],
            }

    def fake_post(url, headers, json, timeout):
        calls["url"] = url
        calls["headers"] = headers
        calls["json"] = json
        calls["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("CHAT_MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("API_GPT_KEY", "gemini-test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(model.requests, "post", fake_post)

    answer = get_answer_from_context_GPT(
        "Context: {information}\nQuestion: {question}",
        ["clinical psychology treats psychological disorders."],
        "What is clinical psychology?",
    )

    assert answer == "Gemini answer from RAG context."
    assert calls["url"].endswith("/models/gemini-2.5-flash:generateContent")
    assert calls["headers"]["x-goog-api-key"] == "gemini-test-key"
    assert calls["json"]["contents"][0]["parts"][0]["text"]
