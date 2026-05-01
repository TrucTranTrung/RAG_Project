import os
from dotenv import load_dotenv

# Lấy đường dẫn đến file hiện tại
load_dotenv()

CHAT_MODEL_PROVIDER = os.getenv("CHAT_MODEL_PROVIDER", "local").lower()


def _clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    return " ".join(text.split())


def _get_openai_client():
    from openai import OpenAI

    return OpenAI(api_key=os.getenv("API_GPT_KEY"))


def get_answer_from_context_LOCAL(
    prompt_template: str,
    information: list[str],
    question: str
) -> str:
    """
    Lightweight local demo responder.
    It does not run an LLM, so it works without API quota and without GPU memory.
    """
    processed_information = _clean_text(" ".join(information))
    if not processed_information:
        return "I do not have enough context in the local knowledge base to answer that demo question."

    max_chars = int(os.getenv("LOCAL_DEMO_MAX_CONTEXT_CHARS", "700"))
    context_preview = processed_information[:max_chars].strip()
    if len(processed_information) > max_chars:
        context_preview += "..."

    return (
        "Demo local answer based on the retrieved context: "
        f"{context_preview}"
    )


def get_answer_from_context_GPT(
    prompt_template: str,
    information: list[str],
    question: str
) -> str:
    if CHAT_MODEL_PROVIDER == "local":
        return get_answer_from_context_LOCAL(prompt_template, information, question)

    try:
        processed_information = _clean_text(" ".join(information))

        prompt = prompt_template.format(
            subject="about psychology",
            information=processed_information,
            question=question
        )

        client = _get_openai_client()
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=150
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("❌ GPT error:", e)
        return ""


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
