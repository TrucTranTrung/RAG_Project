from prompt import prompt_template  
import requests
# from Embedding_Store.Model import *
import os
import json
from dotenv import load_dotenv
# load env
basedir = os.path.abspath(os.path.dirname(__file__))
dotenv_path = os.path.join(basedir, 'config', '.env')

if os.path.exists(dotenv_path):
    print(f"Loading env file from: {dotenv_path}")
    # Tải các biến môi trường từ file .env được chỉ định
    load_dotenv(dotenv_path=dotenv_path) 
else:
    print(f"warning: file .env at {dotenv_path} not found")
    

# embedding_model = initialize_embedding_model(os.getenv("MODEL_NAME_EMBED"))

# --- Gọi mô hình llama3.2 qua Ollama ---
# def call_ollama_llama32(question, context_chunks):
#     try:
#         # print("context:", context_chunks)
#         prompt = prompt_template.format(
#             subject="about programing java",
#             information=(context_chunks),
#             question=question
#         )
        
#         payload = {
#             "model": "llama3.2:3b",
#             "prompt": prompt,
#             "stream": False
#         }

#         response = requests.post("http://localhost:11434/api/generate", json=payload)
#         response.raise_for_status()
#         result = response.json()
#         return result['response'].strip()
#     except Exception as e:
#         print("❌ Error when call ollama:", e)
#         return "Cannot connect to the model. Please check again."



def get_entities_as_string_GEMINI(prompt_template: str, information: str, question: str) -> str:
    """
    Args:
        prompt_template (str): Mẫu prompt có các vị trí để format.
        information (str): Thông tin để đưa vào prompt (ví dụ: reranked_indices).
        question (str): Câu hỏi của người dùng.

    Returns:
        str: Một chuỗi chứa các thực thể đã được xử lý, hoặc một chuỗi rỗng nếu có lỗi.
    """
    api_url = os.getenv("API_GEMINI_ENTITIES")
    if not api_url:
        print("Error: API_GEMINI_ENTITIES not found.")
        return ""

    try:
        processed_information = " ".join(information)
        # Thay thế các ký tự \n và \t bằng dấu cách.
        processed_information = processed_information.replace('\n', ' ').replace('\t', ' ')
        # Loại bỏ các khoảng trắng thừa.
        processed_information = " ".join(processed_information.split())
        # output_path = "query_results.txt"
        # with open(output_path, "w", encoding="utf-8") as f:
        #     f.write(f"Nội dung: {processed_information}\n")
        # Định dạng prompt và tạo payload
        prompt = prompt_template.format(
            subject="about programming java", 
            information=information,
            question=question
        )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        # Gửi yêu cầu POST đến API
        response = requests.post(
            api_url, 
            headers={"Content-Type": "application/json"}, 
            json=payload
        )
        response.raise_for_status() 

        # Xử lý response
        response_json = response.json()
        
        # Trích xuất text 
        text = response_json.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')

        # Xử lý và chuyển đổi text thành list
        output_list = [i.strip().lower() for i in text.replace('[','').replace(']','').replace('"','').split(',') if i.strip()]
        # Chuyển list thành một chuỗi duy nhất
        output_string = ", ".join(output_list)
        
        return output_string

    except requests.exceptions.RequestException as e:
        print(f"Error when calling API: {e}")
        return ""
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Error when processing JSON response: {e}")
        print(f"Response received: {response.text}")
        return ""
    except Exception as e:
        print(f"Error occurred: {e}")
        return ""