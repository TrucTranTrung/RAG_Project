import os
import re
import requests
from dotenv import load_dotenv

# Lấy đường dẫn đến file hiện tại
load_dotenv()

DEFAULT_CHAT_MODEL_PROVIDER = "gemini"


def _clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    return " ".join(text.split())


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "the", "to",
    "what", "when", "where", "which", "who", "why", "with",
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z-]+", text.lower())
    return {word for word in words if len(word) > 2 and word not in _STOPWORDS}


def _sentence_units(text: str) -> list[str]:
    glossary_text = re.sub(r"\s*::\s*", "\n::", text)
    chunks = [chunk.strip(" -•") for chunk in glossary_text.split("\n") if chunk.strip()]
    units: list[str] = []
    for chunk in chunks:
        units.extend(
            sentence.strip(" -•")
            for sentence in re.split(r"(?<=[.!?])\s+", chunk)
            if len(sentence.strip(" -•")) > 20
        )
    return units


def _best_units(question: str, context: str, limit: int = 2) -> list[str]:
    question_keywords = _keywords(question)
    scored_units = []
    for index, unit in enumerate(_sentence_units(context)):
        unit_keywords = _keywords(unit)
        score = len(question_keywords & unit_keywords)
        if score:
            scored_units.append((score, -index, unit))
    return [unit for _, _, unit in sorted(scored_units, reverse=True)[:limit]]


def _readable_unit(unit: str) -> str:
    unit = unit.lstrip(": ")
    unit = re.sub(
        r"^([a-z][a-z -]+?)\s+(a|an)\s+",
        lambda match: f"{match.group(1).capitalize()} is {match.group(2)} ",
        unit,
        count=1,
    )
    return unit


def _comparison_terms(question: str) -> tuple[str, str] | None:
    match = re.search(
        r"\bdifference between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        question.lower().strip(),
    )
    if not match:
        return None
    return match.group(1).strip(" .?"), match.group(2).strip(" .?")


def _find_term_definition(term: str, context: str) -> str | None:
    term_words = term.lower().split()
    best_match = None
    best_score = 0
    for unit in _sentence_units(context):
        unit_lower = unit.lower()
        if term not in unit_lower:
            continue
        score = sum(word in unit_lower for word in term_words)
        score += 2 if "branch of psychology" in unit_lower else 0
        score += 1 if "::" in unit_lower else 0
        if score > best_score:
            best_score = score
            best_match = unit
    return best_match


def _get_openai_client():
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_GPT_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY or API_GPT_KEY.")
    return OpenAI(api_key=api_key)


def _get_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_GPT_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY or API_GPT_KEY.")
    return api_key


def _get_chat_model_provider() -> str:
    return os.getenv("CHAT_MODEL_PROVIDER", DEFAULT_CHAT_MODEL_PROVIDER).lower()


def _build_prompt(prompt_template: str, information: list[str], question: str) -> str:
    processed_information = _clean_text(" ".join(information))
    return prompt_template.format(
        subject="about psychology",
        information=processed_information,
        question=question,
    )


def _get_answer_from_context_OPENAI(prompt: str) -> str:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=float(os.getenv("CHAT_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("CHAT_MAX_TOKENS", "300")),
    )
    return response.choices[0].message.content.strip()


def _get_answer_from_context_GEMINI(prompt: str) -> str:
    base_url = os.getenv(
        "GEMINI_API_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta",
    ).rstrip("/")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    response = requests.post(
        f"{base_url}/models/{model}:generateContent",
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": _get_gemini_api_key(),
        },
        json={
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                    ],
                },
            ],
            "generationConfig": {
                "temperature": float(os.getenv("CHAT_TEMPERATURE", "0.2")),
                "maxOutputTokens": int(os.getenv("CHAT_MAX_TOKENS", "300")),
            },
        },
        timeout=int(os.getenv("GEMINI_TIMEOUT_SECONDS", "120")),
    )
    response.raise_for_status()
    data = response.json()
    parts = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    answer = "".join(part.get("text", "") for part in parts).strip()
    if not answer:
        raise ValueError("Gemini response did not include text content.")
    return answer


def _get_answer_from_context_OLLAMA(prompt: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": float(os.getenv("CHAT_TEMPERATURE", "0.2")),
            },
        },
        timeout=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
    )
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "").strip()


def get_answer_from_context_LOCAL(
    prompt_template: str,
    information: list[str],
    question: str
) -> str:
    """
    Lightweight local extractive responder.
    It does not run an LLM, so it works without API quota and without GPU memory.
    """
    processed_information = _clean_text(" ".join(information))
    if not processed_information:
        return "Not enough information to answer the question."

    comparison_terms = _comparison_terms(question)
    if comparison_terms:
        first_term, second_term = comparison_terms
        first_definition = _find_term_definition(first_term, processed_information)
        second_definition = _find_term_definition(second_term, processed_information)
        if first_definition and second_definition:
            return (
                f"{_readable_unit(first_definition)} "
                f"{_readable_unit(second_definition)} "
                f"In short, {first_term} and {second_term} differ in the main problems "
                "they focus on, based on the retrieved psychology context."
            )

    selected_units = _best_units(question, processed_information)
    if not selected_units:
        return "Not enough information to answer the question."

    return " ".join(_readable_unit(unit) for unit in selected_units)


def get_answer_from_context_GPT(
    prompt_template: str,
    information: list[str],
    question: str
) -> str:
    provider = _get_chat_model_provider()
    if provider in {"extractive", "demo"}:
        return get_answer_from_context_LOCAL(prompt_template, information, question)

    try:
        prompt = _build_prompt(prompt_template, information, question)
        if provider == "gemini":
            return _get_answer_from_context_GEMINI(prompt)
        if provider == "openai":
            return _get_answer_from_context_OPENAI(prompt)
        if provider in {"local", "ollama"}:
            return _get_answer_from_context_OLLAMA(prompt)
        raise ValueError(
            "Unsupported CHAT_MODEL_PROVIDER. Use gemini, openai, local, ollama, or extractive."
        )

    except Exception as e:
        print("❌ Chat model error:", e)
        return (
            "The chat model is not configured or is unavailable. "
            "Set CHAT_MODEL_PROVIDER=gemini with GEMINI_API_KEY/API_GPT_KEY, "
            "set CHAT_MODEL_PROVIDER=openai with OPENAI_API_KEY/API_GPT_KEY, "
            "or set CHAT_MODEL_PROVIDER=local with a running Ollama server."
        )


# def get_entities_as_string_GEMINI(prompt_template: str, information: str, question: str) -> str:
#     """
#     Args:
#         prompt_template (str): Mẫu prompt có các vị trí để format.
#         information (str): Thông tin để đưa vào prompt (ví dụ: reranked_indices).
#         question (str): Câu hỏi của người dùng.

#     Returns:
#         str: Một chuỗi chứa các thực thể đã được xử lý, hoặc một chuỗi rỗng nếu có lỗi.
#     """
#     api_url = os.getenv("API_GEMINI_ENTITIES")
#     if not api_url:
#         print("Error: API_GEMINI_ENTITIES not found.")
#         return ""

#     try:
#         processed_information = " ".join(information)
#         # Thay thế các ký tự \n và \t bằng dấu cách.
#         processed_information = processed_information.replace('\n', ' ').replace('\t', ' ')
#         # Loại bỏ các khoảng trắng thừa.
#         processed_information = " ".join(processed_information.split())
#         # output_path = "query_results.txt"
#         # with open(output_path, "w", encoding="utf-8") as f:
#         #     f.write(f"Nội dung: {processed_information}\n")
#         # Định dạng prompt và tạo payload
#         prompt = prompt_template.format(
#             subject="about psychology", 
#             information=information,
#             question=question
#         )
#         payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
#         # Gửi yêu cầu POST đến API
#         response = requests.post(
#             api_url, 
#             headers={"Content-Type": "application/json"}, 
#             json=payload
#         )
#         response.raise_for_status() 

#         # Xử lý response
#         response_json = response.json()
        
#         # Trích xuất text 
#         text = response_json.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')

#         # Xử lý và chuyển đổi text thành list
#         output_list = [i.strip().lower() for i in text.replace('[','').replace(']','').replace('"','').split(',') if i.strip()]
#         # Chuyển list thành một chuỗi duy nhất
#         output_string = ", ".join(output_list)
        
#         return output_string

#     except requests.exceptions.RequestException as e:
#         print(f"Error when calling API: {e}")
#         return ""
#     except (KeyError, IndexError, json.JSONDecodeError) as e:
#         print(f"Error when processing JSON response: {e}")
#         print(f"Response received: {response.text}")
#         return ""
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         return ""
