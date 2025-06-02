CREATE KEYSPACE IF NOT EXISTS medical_chatbot WITH replication = {
  'class': 'SimpleStrategy',
  'replication_factor': 1
};

USE medical_chatbot;

CREATE TABLE IF NOT EXISTS qa_pairs (
    id UUID PRIMARY KEY,
    qa_id int,
    domain int,
    q_type int,
    question TEXT,
    answer TEXT,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS medical_knowledge (
    id UUID PRIMARY KEY,
    c_id TEXT,
    domain int,
    source int,
    source_spec TEXT,
    creation_year TEXT,
    content TEXT,
    created_at TIMESTAMP
);
