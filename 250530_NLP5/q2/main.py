import re
from sklearn.feature_extraction.text import TfidfVectorizer
import os # 파일 존재 여부 확인을 위해 추가

regex = re.compile('[^a-z ]') # 소문자와 공백을 제외한 모든 문자 제거

documents = []
corpus_file = r"D:\학습\250530\250530_NLP5\q2\text.txt"

try:
    # 텍스트 파일을 읽어와 문서 리스트에 저장
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            lowered_sent = line.rstrip().lower()
            filtered_sent = regex.sub('', lowered_sent)
            if filtered_sent: # 빈 문자열이 아닌 경우에만 추가
                documents.append(filtered_sent)
    
    if not documents:
        raise ValueError("text.txt 파일이 비어있거나 유효한 데이터를 포함하지 않습니다.")
    print(f"'{corpus_file}'에서 {len(documents)}개의 문서를 성공적으로 로드했습니다.")

except (FileNotFoundError, ValueError) as e:
    print(f"경고: '{corpus_file}' 파일을 찾을 수 없거나 ({e}). 예시 데이터를 사용합니다.")
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    documents = [
        "this is a great movie comedy",
        "i love this film acting is superb",
        "the plot was boring and bad",
        "amazing performance wonderful story comedy"
    ]
    print(f"예시 문서 {len(documents)}개를 사용합니다.")


# 1. TfidfVectorizer() 객체를 이용해 TF-IDF Bag of words 문서 벡터를 생성하여 변수 X에 저장하세요.
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(documents)

# 2. 변수 X의 차원을 변수 dim1에 저장하세요.
dim1 = X.shape
# X 변수의 차원을 확인해봅니다.
print(f"\nTF-IDF Bag of words 문서 행렬의 차원: {dim1}")

# 3. 첫 번째 문서의 TF-IDF Bag of words를 vec1 변수에 저장하세요.
vec1 = X[0]
# 첫 번째 문서의 TF-IDF Bag of words를 확인합니다.
print(f"첫 번째 문서의 TF-IDF Bag of words 벡터:\n{vec1}")

# 4. 위에서 생성한 TfidfVectorizer() 객체를 이용해 TF-IDF 기반 Bag of N-grams 문서 벡터를 생성하세요.
# ngram_range=(1, 2)는 unigram과 bigram을 모두 사용하라는 의미입니다.
tfidf_ngram_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
unibigram_X = tfidf_ngram_vectorizer.fit_transform(documents)

# 5. 생성한 TF-IDF 기반 Bag of N-grams 문서 벡터의 차원을 변수 dim2에 저장하세요.
dim2 = unibigram_X.shape
# 문서 벡터의 차원을 확인합니다.
print(f"\nTF-IDF Bag of N-grams 문서 벡터의 차원: {dim2}")