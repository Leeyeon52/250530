import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
from konlpy.tag import Kkma, Okt

# sts-train.tsv 파일에 저장되어 있는 KorSTS 데이터셋을 불러옵니다.
try:
    df = pd.read_table("sts-train.tsv", delimiter='\t', header=0, quoting=3)
    sent = df['sentence1']
except FileNotFoundError:
    print("sts-train.tsv 파일을 찾을 수 없습니다. 예시 데이터를 사용합니다.")
    sent = pd.Series([
        "사무실에서 선풍기 두 대가 돌아가고 있다.",
        "한 남자가 실내에서 스케이트보드를 타고 있다.",
        "한 남자가 파이프에 비눗물을 바르고 있다.",
        "한 여성이 무대에서 춤을 추고 있다.",
        "한 여성이 지시에 따라 요가를 하고 있다."
    ])


# sent 변수에 저장된 첫 5개 문장을 확인해봅니다.
print("---")
print("첫 5개 문장:")
print(sent[:5])

# ---

# 꼬꼬마 형태소 사전을 이용해서 sent 내 문장의 명사를 nouns 리스트에 저장하세요.
kkma = Kkma()
nouns = []
for s in sent:
    nouns.extend(kkma.nouns(s))

# 명사의 종류를 확인해봅니다.
print("\n---")
print("꼬꼬마 명사 추출 결과 (중복 제거):")
print(set(nouns))

# ---

# Open Korean Text 형태소 사전을 이용해서 sent 내 형태소 분석 결과를 pos_results 리스트에 저장하세요.
okt = Okt()
pos_results = []
for s in sent:
    pos_results.append(okt.pos(s))

# 분석 결과를 확인해봅니다.
print("\n---")
print("Okt 형태소 분석 결과 (첫 번째 문장):")
print(pos_results[0])

# ---

# stemming 기반 형태소 분석이 적용된 sent의 두 번째 문장을 stem_pos_results 리스트에 저장하세요.
stem_pos_results = []
if len(sent) > 1:
    stem_pos_results = okt.pos(sent[1], stem=True)
else:
    print("\n두 번째 문장이 존재하지 않아 stemming 분석을 수행할 수 없습니다.")

print("\n---")
print("Okt stemming 기반 두 번째 문장 분석 결과:")
print(stem_pos_results)