import json
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import logging
import re # 정규표현식 모듈 추가

# 로깅 레벨을 DEBUG로 설정하여 더 자세한 정보 확인
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

client = None
db = None

try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client.medical_chatbot
    client.admin.command('ping')
    logging.info("Successfully connected to MongoDB.")
except ConnectionFailure as e:
    logging.error(f"Error connecting to MongoDB: {e}")
    exit()
except Exception as e:
    logging.error(f"An unexpected error occurred during MongoDB connection: {e}")
    exit()

qa_pairs_collection = db.qa_pairs

# MongoDB question 필드에 unique index 생성 (스크립트 첫 실행 시 한번만 주석 풀고 실행)
try:
    qa_pairs_collection.create_index("question", unique=True, background=True)
    logging.info("Unique index on 'question' field created successfully (if not already exists).")
except OperationFailure as e:
    logging.warning(f"Could not create unique index on 'question' field: {e}. It might already exist.")


json_data_root_path = r'D:\학습\250529\09.필수의료 의학지식 데이터\3.개방데이터\1.데이터'

total_inserted_count = 0

try:
    for root, _, files in os.walk(json_data_root_path):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                logging.info(f"Processing file: {json_file_path}")

                data = None
                file_content = None
                for encoding_type in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr', 'latin1']:
                    try:
                        with open(json_file_path, 'r', encoding=encoding_type, errors='ignore') as f:
                            file_content = f.read()
                            data = json.loads(file_content)
                        logging.info(f"Successfully loaded '{file}' with {encoding_type} encoding.")
                        break
                    except json.JSONDecodeError as jde:
                        logging.warning(f"Could not decode JSON from '{file}' with {encoding_type} encoding. Error: {jde}. Trying next.")
                        if logging.getLogger().level == logging.DEBUG:
                             logging.debug(f"Failed content snippet: {file_content[:500] if file_content else 'N/A'}...")
                        data = None
                    except Exception as e:
                        logging.error(f"An unexpected error occurred while reading {json_file_path} with {encoding_type}: {e}. Skipping encodings.")
                        data = None
                        break

                if data is None:
                    logging.error(f"Skipping '{file}': Failed to decode JSON with any tried encoding.")
                    continue

                documents_to_insert = []
                
                if isinstance(data, dict):
                    if "content" in data:
                        # --- 대용량 content 필드 분할 로직 시작 ---
                        content_text = data["content"]
                        # 정규표현식을 사용하여 "1. ", "2. ", ... 패턴으로 문장 분리
                        # 첫 번째 항목도 포함하기 위해 (?:^|\s)를 사용하여 줄 시작 또는 공백 뒤의 숫자를 찾습니다.
                        # re.split()은 구분자를 결과 리스트에 포함하지 않으므로,
                        # lookahead assertion을 사용하여 숫자를 기준으로 분리하되, 숫자를 포함하도록 합니다.
                        
                        # 예시: '1. 첫 번째 내용. 2. 두 번째 내용.' -> ['1. 첫 번째 내용.', '2. 두 번째 내용.']
                        split_contents = re.split(r'(?=\d+\.\s)', content_text)
                        
                        # re.split()이 빈 문자열을 반환하는 경우를 처리 (예: 첫 항목이 구분자로 시작하지 않는 경우)
                        # 또는 맨 앞의 빈 문자열 제거
                        split_contents = [s.strip() for s in split_contents if s.strip()]

                        if not split_contents: # 분할된 내용이 없다면 전체를 하나의 문서로
                            split_contents = [content_text]

                        for i, chunk in enumerate(split_contents):
                            # 질문 생성: 파일 메타데이터 + 청크 번호/내용 일부
                            # 청크 시작 부분에서 첫 50자 정도를 따와서 질문에 포함하거나,
                            # 더 일반적인 질문 패턴을 사용합니다.
                            # 여기서는 'source_spec'과 'c_id'를 기반으로 일반 질문을 만들고,
                            # 청크 내용을 답변으로 넣습니다.
                            question_text = f"'{data.get('source_spec', '알 수 없음')}'의 '{data.get('c_id', '알 수 없음')}' 데이터 중 {i+1}번째 항목에 대한 정보를 알려줘."
                            # 또는 더 간결하게: f"{data.get('source_spec', '')} {data.get('c_id', '')} - {chunk[:50]}... 에 대해 알려줘."
                            
                            # 답변은 현재 청크
                            answer_text = chunk
                            
                            documents_to_insert.append({
                                "question": question_text,
                                "answer": answer_text,
                                "c_id": data.get("c_id"),
                                "domain": data.get("domain"),
                                "source": data.get("source"),
                                "source_spec": data.get("source_spec"),
                                "creation_year": data.get("creation_year"),
                                "original_file": file, # 어떤 파일에서 추출되었는지 저장
                                "chunk_index": i + 1 # 몇 번째 청크인지 기록
                            })
                        logging.info(f"Generated {len(documents_to_insert)} documents from '{file}' by splitting 'content' field.")
                        # --- 대용량 content 필드 분할 로직 끝 ---

                    # 'question'과 'answer' 필드를 가진 단일 객체 처리 (예: 이전 소아청소년과 파일)
                    elif "question" in data and "answer" in data:
                        documents_to_insert.append({
                            "question": data["question"],
                            "answer": data["answer"],
                            "c_id": data.get("c_id"), # 메타데이터 추가
                            "domain": data.get("domain"),
                            "source": data.get("source"),
                            "source_spec": data.get("source_spec"),
                            "creation_year": data.get("creation_year"),
                            "original_file": file
                        })
                        logging.info(f"Generated 1 document from '{file}' based on 'question' and 'answer' fields.")
                    else:
                        logging.warning(f"Skipping '{file}': No 'content' or 'question/answer' fields found in dictionary format. Keys found: {data.keys()}")
                
                elif isinstance(data, list):
                    # 리스트 형태의 JSON 파일 처리 (각 리스트 아이템이 Q&A 또는 Content 객체)
                    for item in data:
                        if isinstance(item, dict):
                            if "question" in item and "answer" in item:
                                documents_to_insert.append({
                                    "question": item["question"],
                                    "answer": item["answer"],
                                    "c_id": item.get("c_id"), # 메타데이터 추가
                                    "domain": item.get("domain"),
                                    "source": item.get("source"),
                                    "source_spec": item.get("source_spec"),
                                    "creation_year": item.get("creation_year"),
                                    "original_file": file
                                })
                            elif "content" in item:
                                # 리스트 안의 객체가 content 필드만 가진 경우
                                content_text = item["content"]
                                split_contents = re.split(r'(?=\d+\.\s)', content_text)
                                split_contents = [s.strip() for s in split_contents if s.strip()]

                                if not split_contents:
                                    split_contents = [content_text]
                                
                                for i, chunk in enumerate(split_contents):
                                    question_text = f"'{item.get('source_spec', '알 수 없음')}'의 '{item.get('c_id', '알 수 없음')}' 데이터 중 {i+1}번째 항목에 대한 정보를 알려줘."
                                    documents_to_insert.append({
                                        "question": question_text,
                                        "answer": chunk,
                                        "c_id": item.get("c_id"),
                                        "domain": item.get("domain"),
                                        "source": item.get("source"),
                                        "source_spec": item.get("source_spec"),
                                        "creation_year": item.get("creation_year"),
                                        "original_file": file,
                                        "chunk_index": i + 1
                                    })
                                logging.info(f"Generated {len(documents_to_insert)} documents from list item in '{file}' by splitting 'content'.")
                            else:
                                logging.warning(f"Skipping invalid item in list in '{file}': {item.keys()}")
                        else:
                            logging.warning(f"Skipping non-dictionary item in list in '{file}': {type(item)}")
                    if not documents_to_insert:
                        logging.warning(f"No valid Q&A documents found within list in '{file}'.")
                else:
                    logging.warning(f"Skipping '{file}': Unexpected JSON format. Expected dict or list. Type found: {type(data)}")

                if documents_to_insert:
                    try:
                        for doc in documents_to_insert:
                            qa_pairs_collection.update_one(
                                {"question": doc["question"]},
                                {"$set": doc},
                                upsert=True
                            )
                        total_inserted_count += len(documents_to_insert)
                        logging.info(f"Upserted {len(documents_to_insert)} documents from {file}.")
                    except OperationFailure as e:
                        logging.error(f"MongoDB operation failed for {file}: {e}")
                else:
                    logging.info(f"No valid Q&A documents generated from {file}.")

except FileNotFoundError:
    logging.error(f"Error: The root path '{json_data_root_path}' was not found.")
except OperationFailure as e:
    logging.error(f"MongoDB operation failed for root path: {e}")
except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")
finally:
    if client:
        client.close()
        logging.info("MongoDB connection closed.")