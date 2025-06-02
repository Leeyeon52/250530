from cassandra.cluster import Cluster
from uuid import uuid4
from datetime import datetime

class CassandraDB:
    def __init__(self):
        self.cluster = Cluster(['localhost'])
        self.session = self.cluster.connect('medical_chatbot')

    def insert_qa(self, qa_id, domain, q_type, question, answer):
        self.session.execute("""
            INSERT INTO qa_pairs (id, qa_id, domain, q_type, question, answer, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (uuid4(), qa_id, domain, q_type, question, answer, datetime.utcnow()))

    def insert_knowledge(self, c_id, domain, source, source_spec, creation_year, content):
        self.session.execute("""
            INSERT INTO medical_knowledge (id, c_id, domain, source, source_spec, creation_year, content, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (uuid4(), c_id, domain, source, source_spec, creation_year, content, datetime.utcnow()))
