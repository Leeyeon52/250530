import requests
import json

# API 엔드포인트 URL
API_URL = "http://localhost:5000/ask"

def ask_chatbot(question_text):
    """
    챗봇 API에 질문을 보내고 응답을 받습니다.
    """
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "question": question_text
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # HTTP 오류가 발생하면 예외 발생

        result = response.json()
        if "answer" in result:
            print(f"질문: {question_text}")
            print(f"답변: {result['answer']}")
        elif "error" in result:
            print(f"오류: {result['error']}")
        else:
            print(f"알 수 없는 응답 형식: {result}")

    except requests.exceptions.ConnectionError:
        print(f"오류: 챗봇 API ({API_URL})에 연결할 수 없습니다. API 서버가 실행 중인지 확인하세요.")
    except requests.exceptions.RequestException as e:
        print(f"요청 중 오류 발생: {e}")
    except json.JSONDecodeError:
        print(f"오류: 서버 응답이 유효한 JSON 형식이 아닙니다.")
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")

if __name__ == "__main__":
    # 테스트 질문 (MongoDB에 있는 데이터 중 일부를 사용해 보세요)
    # 예시:
    # "피로"
    # "세브란스병원 54099_1 데이터 중 1번째 항목에 대한 정보를 알려줘."
    # "급성 신부전 증상은 무엇인가요?"
    # "뇌경색의 주요 증상은 무엇입니까?"

    test_question = "피로" 
    ask_chatbot(test_question)

    # 다른 질문을 시도하려면 아래 주석을 풀고 사용하세요
    # print("\n--- 다른 질문 ---")
    # ask_chatbot("뇌경색")