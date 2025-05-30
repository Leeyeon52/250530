import re
from sklearn.feature_extraction.text import CountVectorizer
import os

regex = re.compile('[^a-z ]') # 소문자와 공백을 제외한 모든 문자 제거

documents = []
corpus_file = r"D:\학습\250530\250530_NLP5\q1\text.txt"

try:
    # 텍스트 파일을 읽어와 문서 리스트에 저장
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. documents 리스트에 리뷰 데이터를 저장하세요.
            # regex 변수에 주어진 정규표현식을 사용하여 전처리된 데이터를 저장하세요.
            processed_line = regex.sub('', line.lower()).strip() # 소문자 변환 및 양 끝 공백 제거
            if processed_line: # 빈 문자열이 아닌 경우에만 추가
                documents.append(processed_line)
    
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


# 2. CountVectorizer() 객체를 이용해 Bag of words 문서 벡터를 생성하여 변수 X에 저장하세요.
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# 3. 변수 X의 차원을 변수 dim에 저장하세요.
dim = X.shape
# X 변수의 차원을 확인해봅니다.
print(f"\nBag of words 문서 행렬의 차원: {dim}")

# 4. 위에서 생성한 CountVectorizer() 객체에서 첫 10개의 칼럼이 의미하는 단어를 words_feature 변수에 저장하세요.
# `get_feature_names_out()`는 scikit-learn 0.24 버전부터 `get_feature_names()`를 대체합니다.
words_feature = vectorizer.get_feature_names_out()[:10]
# CountVectorizer() 객체의 첫 10개 칼럼이 의미하는 단어를 확인해봅니다.
print(f"첫 10개 칼럼의 단어: {words_feature}")

# 5. 단어 "comedy"를 의미하는 칼럼의 인덱스 값을 idx 변수에 저장하세요.
# `.get()` 메소드를 사용하여 단어가 존재하지 않을 경우 None을 반환하도록 합니다.
idx = vectorizer.vocabulary_.get("comedy")
# 단어 "comedy"의 인덱스를 확인합니다.
if idx is not None:
    print(f"단어 'comedy'의 칼럼 인덱스: {idx}")
else:
    print(f"단어 'comedy'를 찾을 수 없습니다. (전처리 과정에서 제외되었거나 문서에 없음)")


# 6. 첫 번째 문서의 Bag of words 벡터를 vec1 변수에 저장하세요.
vec1 = X[0]
# 첫 번째 문서의 Bag of words 벡터를 확인합니다.
print(f"첫 번째 문서의 Bag of words 벡터:\n{vec1}")