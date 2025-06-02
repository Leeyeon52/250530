import json
from cassandra_connector import CassandraDB

def insert_data():
    db = CassandraDB()

    # Q&A 데이터 삽입
    with open("sample_qa.json", "r", encoding="utf-8") as f:
        qa_items = json.load(f)
        for item in qa_items:
            db.insert_qa(item)

    # 의학 지식 데이터 삽입
    with open("sample_knowledge.json", "r", encoding="utf-8") as f:
        knowledge_items = json.load(f)
        for item in knowledge_items:
            db.insert_knowledge(item)

if __name__ == "__main__":
    insert_data()
CREATE KEYSPACE medical_chatbot WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};
