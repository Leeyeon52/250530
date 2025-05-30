import os
from soynlp.noun import LRNounExtractor_v2

sent = '트와이스 아이오아이 좋아여 tt가 저번에 1위 했었죠?'

corpus_path = r'D:\학습\250530\250530_NLP4\q2\q2_articles\articles.txt'
train_data = []

# DoublespaceLineCorpus 대신 직접 파일 읽기
if os.path.exists(corpus_path):
    try:
        # 'articles.txt' 파일을 읽고, 빈 줄을 기준으로 문장을 분리합니다.
        # 인코딩은 'utf-8'이 일반적이지만, 파일에 따라 'euc-kr' 등을 시도할 수 있습니다.
        with open(corpus_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            # 빈 줄 두 개를 기준으로 문장을 분리하고, 각 문장의 양 끝 공백을 제거
            # soynlp의 DoublespaceLineCorpus처럼 동작하도록 함
            sentences = [s.strip() for s in raw_text.split('\n\n') if s.strip()]
            train_data = sentences

        if len(train_data) == 0:
            print(f"경고: '{corpus_path}' 파일은 존재하지만 내용이 비어 있거나 올바른 형식의 문장을 포함하지 않습니다.")
            print("예시 데이터를 사용하여 학습을 진행합니다.")
            train_data = [
                "대한민국은 동아시아의 한반도 남부에 있는 나라이다.",
                "수도는 서울이며, 공용어는 한국어이다.",
                "경제 성장과 함께 정보통신기술 분야에서 두각을 나타내고 있다.",
                "K-POP과 드라마 등 한류 문화가 전 세계적으로 인기를 끌고 있다."
            ]
        print("학습 문서의 개수: %d" % (len(train_data)))

    except Exception as e:
        print(f"경고: '{corpus_path}' 파일을 읽는 중 예기치 않은 오류가 발생했습니다: {e}")
        print("예시 데이터를 사용하여 학습을 진행합니다.")
        train_data = [
            "대한민국은 동아시아의 한반도 남부에 있는 나라이다.",
            "수도는 서울이며, 공용어는 한국어이다.",
            "경제 성장과 함께 정보통신기술 분야에서 두각을 나타내고 있다.",
            "K-POP과 드라마 등 한류 문화가 전 세계적으로 인기를 끌고 있다."
        ]
        print("예시 학습 문서의 개수: %d" % (len(train_data)))
else:
    print(f"경고: '{corpus_path}' 파일을 찾을 수 없습니다. (현재 작업 디렉토리: {os.getcwd()})")
    print("예시 데이터를 사용하여 학습을 진행합니다.")
    train_data = [
        "대한민국은 동아시아의 한반도 남부에 있는 나라이다.",
        "수도는 서울이며, 공용어는 한국어이다.",
        "경제 성장과 함께 정보통신기술 분야에서 두각을 나타내고 있다.",
        "K-POP과 드라마 등 한류 문화가 전 세계적으로 인기를 끌고 있다."
    ]
    print("예시 학습 문서의 개수: %d" % (len(train_data)))


# LRNounExtractor_v2 객체를 이용해 train_data에서 명사로 추정되는 단어를 nouns 변수에 저장하세요.
noun_extractor = LRNounExtractor_v2()
nouns = noun_extractor.train_extract(train_data)

# 생성된 명사의 개수를 확인해봅니다.
print(f"\n생성된 명사의 개수: {len(nouns)}개")

# 생성