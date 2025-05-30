# 경고문을 제거합니다.
import warnings
warnings.filterwarnings(action='ignore')

import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
from sklearn.feature_extraction.text import TfidfVectorizer

sent1 = ["I first saw this movie when I was a little kid and fell in love with it at once."]
sent2 = ["Despite having 6 different directors, this fantasy hangs together remarkably well."]

# bow_models.pkl 파일 경로 설정 (이전 답변에서 사용했던 정확한 경로를 다시 사용)
# 파일이 실제로 있는 경로로 수정해 주세요.
pkl_file_path = r'D:\학습\250530\250530_NLP5\q3\bow_models.pkl'

# --- File Generation Check ---
# This block runs ONLY if the .pkl file does NOT exist.
# It creates the file and then EXITS, requiring a manual restart.
if not os.path.exists(pkl_file_path):
    print(f"경고: '{pkl_file_path}' 파일을 찾을 수 없습니다. 임시로 생성합니다.")
    print("이 임시 생성은 예시 데이터를 사용하여 모델을 만듭니다. 실제 IMDB 데이터와 다를 수 있습니다.")

    # 임시 documents (실제 IMDB 데이터가 필요)
    temp_documents = [
        "this is a great movie comedy",
        "i love this film acting is superb",
        "the plot was boring and bad",
        "amazing performance wonderful story comedy",
        "i first saw this movie when i was a little kid and fell in love with it at once",
        "despite having different directors this fantasy hangs together remarkably well",
        "the acting was terrible and the story made no sense",
        "highly recommend this amazing movie a true masterpiece"
    ]

    temp_vectorizer = TfidfVectorizer()
    # TfidfVectorizer를 학습(fit)하고 변환(transform)하여 temp_X에 저장합니다.
    # 이렇게 해야 vectorizer가 'fit'된 상태로 pickle에 저장됩니다.
    temp_X = temp_vectorizer.fit_transform(temp_documents)

    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(pkl_file_path), exist_ok=True)

    with open(pkl_file_path, 'wb') as f:
        pickle.dump((temp_vectorizer, temp_X), f)
    print(f"'{pkl_file_path}' 파일을 임시로 성공적으로 생성했습니다.")
    print("----------------------------------------------------------------------")
    print("--- 중요: 스크립트를 수동으로 종료하고, 다시 실행해야 합니다. ---")
    print("----------------------------------------------------------------------")
    exit() # 파일 생성 후 즉시 종료

# --- Main Logic (runs only if the .pkl file already exists) ---
try:
    with open(pkl_file_path, 'rb') as f:
        # 저장된 모델을 불러와 객체와 벡터를 각각 vectorizer와 X에 저장하세요.
        vectorizer, X = pickle.load(f)
    print(f"'{pkl_file_path}'에서 vectorizer와 X를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print(f"오류: '{pkl_file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인하거나 파일을 먼저 생성해주세요.")
    exit()
except Exception as e:
    print(f"오류: pickle 파일을 불러오는 중 예기치 않은 오류가 발생했습니다: {e}")
    exit()

# sent1, sent2 문장을 vectorizer 객체의 transform() 함수를 이용해 변수 vec1, vec2에 저장합니다.
vec1 = vectorizer.transform(sent1)
vec2 = vectorizer.transform(sent2)
print(f"sent1의 TF-IDF 벡터 형태: {vec1.shape}")
print(f"sent2의 TF-IDF 벡터 형태: {vec2.shape}")

# vec1과 vec2의 코사인 유사도를 변수 sim1에 저장합니다.
sim1 = cosine_similarity(vec1, vec2)[0][0]
print(f"\n'{sent1[0]}'와 '{sent2[0]}'의 코사인 유사도 (sim1): {sim1}")

# vec1과 행렬 X의 첫 번째 문서 벡터 간 코사인 유사도를 변수 sim2에 저장합니다.
sim2 = cosine_similarity(vec1, X[0])[0][0]
print(f"'{sent1[0]}'와 학습 문서의 첫 번째 문서 간 코사인 유사도 (sim2): {sim2}")