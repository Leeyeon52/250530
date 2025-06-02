from flask import Flask, request, jsonify
import logging

# MongoDB 임포트 추가
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# MongoDB 클라이언트 및 DB 변수
client = None
db = None

# MongoDB 연결 함수
def connect_to_mongodb():
    global client, db
    try:
        logging.info("Attempting to connect to MongoDB...")
        # MongoDB 클라이언트 생성 (기본값: localhost:27017)
        client = MongoClient('mongodb://localhost:27017/')
        # 'medical_chatbot' 데이터베이스 선택
        db = client.medical_chatbot
        # 연결 테스트 (실제로 DB에서 명령을 실행하여 연결 확인)
        client.admin.command('ping') 
        logging.info("Successfully connected to MongoDB.")
    except ConnectionFailure as e:
        logging.error(f"Could not connect to MongoDB: {e}")
        client = None
        db = None
    except Exception as e:
        logging.error(f"An unexpected error occurred during MongoDB connection: {e}")
        client = None
        db = None

# 애플리케이션 시작 시 MongoDB 연결 시도 (애플리케이션 컨텍스트 내에서)
with app.app_context():
    connect_to_mongodb()

@app.route('/')
def home():
    logging.info("Root endpoint accessed.")
    return "안녕하세요! 챗봇 API가 실행 중입니다. `/ask` 엔드포인트로 POST 요청을 보내 질문해보세요."

@app.route("/ask", methods=["POST"])
def ask():
    print("DEBUG: /ask 엔드포인트에 요청이 들어왔습니다.") # 디버그용 print문 추가

    # MongoDB DB 객체 확인
    # 'db' 객체의 진릿값 평가 방식을 'None'과의 명시적 비교로 수정
    if db is None: 
        logging.error("MongoDB connection is not established. Cannot process request.")
        return jsonify({"answer": "서버 내부 오류: 데이터베이스에 연결할 수 없습니다."}), 500

    try:
        # 이전에 강제로 에러를 발생시키기 위해 추가했다면, 지금은 주석 처리되어 있습니다.
        # 강제 에러 테스트를 다시 하고 싶다면 아래 줄의 주석을 제거하세요.
        # result = 1 / 0 
        
        user_question = request.json.get("question")
        print(f"DEBUG: 수신된 질문: {user_question}") # 디버그용 print문 추가

        if not user_question:
            return jsonify({"error": "질문 내용이 없습니다."}), 400

        logging.info(f"Received question: {user_question}")

        # MongoDB에서 질문 검색 로직
        # 'qa_pairs' 컬렉션에서 질문을 검색합니다.
        # $regex를 사용하여 부분 문자열 일치 검색을 수행합니다 (대소문자 무시를 위해 'i' 옵션).
        # 실제 운영 환경에서는 인덱스(text index)를 사용하거나 Atlas Search 같은 기능을 고려해야 합니다.
        query = {
            "question": { "$regex": user_question, "$options": "i" }
        }
        print(f"DEBUG: MongoDB 쿼리: {query}") # 디버그용 print문 추가
        
        # find_one은 첫 번째 일치하는 문서를 반환합니다.
        found_doc = db.qa_pairs.find_one(query) 
        print(f"DEBUG: MongoDB 검색 결과 (found_doc): {found_doc}") # 디버그용 print문 추가

        found_answer = None
        if found_doc:
            found_answer = found_doc.get('answer') # 문서에서 'answer' 필드 가져오기

        if found_answer:
            logging.info(f"Answer found for '{user_question}': {found_answer[:50]}...")
            print(f"DEBUG: 찾은 답변: {found_answer}") # 디버그용 print문 추가
            return jsonify({"answer": found_answer})
        else:
            logging.info(f"No answer found for '{user_question}'.")
            print("DEBUG: 답변을 찾지 못했습니다.") # 디버그용 print문 추가
            return jsonify({"answer": "죄송합니다. 해당 질문에 대한 답변을 찾을 수 없습니다."})

    except KeyError:
        logging.error("Invalid JSON format: 'question' key not found.")
        print("DEBUG: KeyError 발생: 'question' 키 없음") # 디버그용 print문 추가
        return jsonify({"error": "요청 JSON 형식 오류: 'question' 키가 필요합니다."}), 400
    except Exception as e:
        logging.error(f"An error occurred during API request: {e}")
        print(f"DEBUG: API 요청 중 예외 발생: {e}") # 디버그용 print문 추가
        return jsonify({"answer": f"서버 내부 오류가 발생했습니다: {e}"}), 500

if __name__ == '__main__':
    logging.info("Attempting to connect to MongoDB...")
    try:
        # 이미 connect_to_mongodb()에서 연결을 시도하므로 여기서는 간단히 확인
        # 'client'와 'db' 객체의 진릿값 평가 방식을 'None'과의 명시적 비교로 수정
        if client is not None and db is not None: 
            client.admin.command('ping') # MongoDB 연결 확인 명령어
            logging.info("Successfully connected to MongoDB (verified in __main__).")
        else:
            logging.error("MongoDB connection is not available. Please check connection logs.")
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB in __main__ check: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)