# -*- coding: utf-8 -*-
import random
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from numpy import sqrt, dot
import os # 파일 존재 여부 확인을 위해 추가

random.seed(10)

doc1_text = "homelessness has been an issue for years but never a plan to help those on the street that were once considered human who did everything from going to school work or vote for the matter"
doc2_text = "it may have ends that do not tie together particularly well but it is still a compelling enough story to stick with"

# doc1과 doc2를 리스트 형태로 변경 (infer_vector는 단어 리스트를 기대하므로 split() 적용)
doc1 = doc1_text.split()
doc2 = doc2_text.split()


# 데이터를 불러오는 함수입니다.
def load_data(filepath):
    regex = re.compile('[^a-z ]')

    gensim_input = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                lowered_sent = line.rstrip().lower()
                # TaggedDocument는 words 인자로 단어 리스트를 받으므로 split() 적용
                filtered_sent_words = regex.sub('', lowered_sent).split()
                if filtered_sent_words: # 빈 리스트가 아닌 경우에만 추가
                    tagged_doc = TaggedDocument(filtered_sent_words, [idx])
                    gensim_input.append(tagged_doc)
        if not gensim_input:
            raise ValueError(f"'{filepath}' 파일이 비어있거나 유효한 데이터를 포함하지 않습니다.")
        print(f"load_data(): '{filepath}'에서 {len(gensim_input)}개의 문서를 성공적으로 로드했습니다.")
    except (FileNotFoundError, ValueError) as e:
        print(f"load_data(): 경고: '{filepath}' 파일을 찾을 수 없거나 ({e}). 예시 데이터를 사용합니다.")
        print(f"현재 작업 디렉토리: {os.getcwd()}")
        # 파일이 없을 때 사용할 예시 Doc2Vec TaggedDocument 데이터
        gensim_input = [
            TaggedDocument(words="this is a great movie comedy".split(), tags=[0]),
            TaggedDocument(words="i love this film acting is superb".split(), tags=[1]),
            TaggedDocument(words="the plot was boring and bad".split(), tags=[2]),
            TaggedDocument(words="amazing performance wonderful story comedy".split(), tags=[3]),
            TaggedDocument(words=doc1_text.split(), tags=[4]), # doc1 내용 포함
            TaggedDocument(words=doc2_text.split(), tags=[5])  # doc2 내용 포함
        ]
        print(f"load_data(): 예시 문서 {len(gensim_input)}개를 사용합니다.")

    return gensim_input

def cal_cosine_sim(v1, v2):
    # 벡터 간 코사인 유사도를 계산해 주는 함수를 완성합니다.
    dot_product = dot(v1, v2)
    magnitude_v1 = sqrt(dot(v1, v1))
    magnitude_v2 = sqrt(dot(v2, v2))

    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0

    cosine_similarity_val = dot_product / (magnitude_v1 * magnitude_v2)
    return cosine_similarity_val

# doc2vec 모델을 documents 리스트를 이용해 학습하세요.
documents = load_data("text.txt")

# Doc2Vec 모델 초기화 (documents 인자를 제거하고 build_vocab을 명시적으로 호출)
# window는 2, vector_size는 50, epochs는 5를 임의로 설정
model = Doc2Vec(vector_size=50, window=2, epochs=5, workers=4)

# 모델의 어휘집을 먼저 구축합니다.
print("\nDoc2Vec 모델 어휘집 구축 시작...")
model.build_vocab(documents)
print("Doc2Vec 모델 어휘집 구축 완료.")

# 어휘집 구축 후 모델을 학습합니다.
print("Doc2Vec 모델 학습 시작...")
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
print("Doc2Vec 모델 학습 완료.")


# 학습된 모델을 이용해 doc1과 doc2에 들어있는 문서의 임베딩 벡터를 생성하여 각각 변수 vector1과 vector2에 저장하세요.
# infer_vector 함수는 단어 리스트를 인자로 받으므로, doc1_text와 doc2_text를 split()하여 전달합니다.
vector1 = model.infer_vector(doc1)
vector2 = model.infer_vector(doc2)
print(f"doc1 임베딩 벡터 생성 완료 (크기: {len(vector1)})")
print(f"doc2 임베딩 벡터 생성 완료 (크기: {len(vector2)})")


# vector1과 vector2의 코사인 유사도를 변수 sim에 저장하세요.
sim = cal_cosine_sim(vector1, vector2)
# 계산한 코사인 유사도를 확인합니다.
print(f"\ndoc1과 doc2의 코사인 유사도: {sim}")