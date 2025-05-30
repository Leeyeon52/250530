import random
from collections import defaultdict

random.seed(10)

data = ['this is a dog', 'this is a cat', 'this is my horse', 'my name is elice', 'my name is hank']

def count_unigram(docs):
    unigram_counter = defaultdict(int) # defaultdict를 사용하면 키가 없을 때 자동으로 0으로 초기화
    # docs에서 발생하는 모든 unigram의 빈도수를 딕셔너리 unigram_counter에 저장하여 반환하세요.
    for doc in docs:
        words = doc.split()
        for word in words:
            unigram_counter[word] += 1
    return unigram_counter

def count_bigram(docs):
    bigram_counter = defaultdict(int) # defaultdict를 사용하면 키가 없을 때 자동으로 0으로 초기화
    # docs에서 발생하는 모든 bigram의 빈도수를 딕셔너리 bigram_counter에 저장하여 반환하세요.
    for doc in docs:
        words = doc.split()
        # bigram은 (현재 단어, 다음 단어) 형태이므로, 리스트 길이 - 1까지 반복
        for i in range(len(words) - 1):
            bigram = (words[i], words[i+1])
            bigram_counter[bigram] += 1
    return bigram_counter

def cal_prob(sent, unigram_counter, bigram_counter):
    words = sent.split()
    result = 1.0
    
    # 문장의 첫 단어는 항상 확률 1로 시작한다고 가정합니다. (언어 모델의 일반적인 시작)
    # 실제로는 <SOS> (Start Of Sentence) 토큰을 사용하기도 합니다.
    
    # bigram 확률 P(w_i | w_{i-1}) 계산
    # 즉, P(w1, w2, ..., wn) = P(w1) * P(w2|w1) * P(w3|w2) * ... * P(wn|w_{n-1})
    # 여기서는 P(w1)을 1로 가정하고 P(w_i|w_{i-1})만 계산합니다.
    
    if len(words) < 2:
        # 단일 단어 문장이거나 빈 문장의 경우 bigram 확률 계산 불가
        # 예시 데이터와 "this is elice"는 최소 2단어 이상이므로 이 조건에 해당하지 않을 가능성이 높음
        if words[0] in unigram_counter:
            return 1.0 # 첫 단어가 unigram에 있으면 1 (간단한 모델 가정)
        else:
            return 0.0
            
    for i in range(len(words) - 1):
        current_word = words[i]
        next_word = words[i+1]
        
        bigram = (current_word, next_word)
        
        bigram_freq = bigram_counter[bigram]
        current_word_freq = unigram_counter[current_word]
        
        # 조건부 확률 계산: P(다음 단어 | 현재 단어) = count(현재 단어, 다음 단어) / count(현재 단어)
        if current_word_freq == 0:
            # 현재 단어가 unigram_counter에 없으면 조건부 확률은 0으로 간주 (zero probability problem)
            # 실제 언어 모델에서는 Smoothing (예: 라플라스 스무딩)을 적용하여 0이 되는 것을 방지합니다.
            prob = 0.0
        else:
            prob = bigram_freq / current_word_freq
            
        result *= prob
        
    return result

# 주어진 data를 이용해 unigram 빈도수, bigram 빈도수를 구합니다.
unigram_counts_result = count_unigram(data)
bigram_counts_result = count_bigram(data)

print(f"Unigram 빈도수: {unigram_counts_result}")
print(f"Bigram 빈도수: {bigram_counts_result}")

# "this is elice" 문장의 발생 확률을 계산해봅니다.
sentence_to_check = "this is elice"
probability = cal_prob(sentence_to_check, unigram_counts_result, bigram_counts_result)

print(f"\n문장 '{sentence_to_check}'의 발생 확률: {probability}")