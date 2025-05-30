# 경고문을 무시합니다.
import warnings
warnings.filterwarnings(action='ignore')

import os
# GPU 사용 비활성화 및 oneDNN 최적화 비활성화
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pickle
import tensorflow as tf
import numpy as np


# 학습된 모델을 불러오는 함수입니다.
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim), 
        
        # 각 시점별 문자예측을 위한 LSTM 구조입니다.
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True, # 이 레이어가 상태를 가집니다.
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
        ])
    return model

# 학습된 모델에서 문장을 생성하는 함수입니다.
def generate_text(model, start_string, num_generate=1000, temperature=1.0):
    # 예측할 문자 혹은 문자열의 정수형 인덱스로 변환
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0) # 배치 차원 추가

    text_generated = []

    # === 수정된 부분 시작 ===
    # 'Sequential' 객체에는 'reset_states' 메소드가 없으므로,
    # 상태를 가지는 (stateful) LSTM 레이어의 상태를 직접 리셋합니다.
    # 모델 내부에 LSTM 레이어가 하나만 있다고 가정하고 index 1 (Embedding 다음)에 접근합니다.
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.LSTM) and layer.stateful:
            layer.reset_states()
            break # 첫 번째 stateful LSTM 레이어만 리셋하고 종료 (필요하다면 모든 LSTM 레이어에 적용 가능)
    # === 수정된 부분 끝 ===

    for i in range(num_generate):
        predictions = model(input_eval)
        
        # 다음 단어 예측을 위한 로짓을 추출하고, temperature 적용 (다양성 조절)
        predictions = predictions[:, -1, :] / temperature 
        
        # 다음 발생확률이 제일 높은 문자로 예측 (np.argmax 대신 tf.random.categorical 사용)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0][0].numpy()
        
        input_eval = tf.expand_dims([predicted_id], 0) # 다음 입력으로 사용하기 위해 차원 확장
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


# 기존 학습한 모델의 구조를 불러옵니다.
# 예측을 위해 batch_size는 1로 조절되었습니다.
vocab_size_rnn = 65
embedding_dim_rnn = 256
rnn_units_rnn = 1024

model = build_model(vocab_size_rnn, embedding_dim_rnn, rnn_units_rnn, batch_size=1)

# 체크포인트 경로 설정 (실제 .ckpt 파일이 있는 경로로 수정하세요)
checkpoint_dir = './checkpoints' 
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

if latest_checkpoint:
    # model.load_weights()을 이용해 데이터를 불러오세요.
    model.load_weights(latest_checkpoint)
    # 모델의 입력 형태를 명시적으로 빌드합니다. (여기서 batch_size를 설정합니다.)
    model.build(tf.TensorShape([1, None])) 
    print(f"모델 가중치를 '{latest_checkpoint}'에서 성공적으로 불러왔습니다.")
else:
    print(f"경고: './checkpoints'에서 학습된 모델 체크포인트를 찾을 수 없습니다.")
    print("모델 가중치를 로드하지 못했습니다. 생성된 텍스트는 의미가 없을 수 있습니다.")
    print("TensorFlow 튜토리얼 등을 참고하여 모델을 학습하고 체크포인트를 저장해야 합니다.")


# char2idx, idx2char는 주어진 문자를 정수 인덱스로 매핑하는 딕셔너리 입니다.
word_index_path = r'D:\학습\250530\250530_NLP5\q6\word_index.pkl' # word_index.pkl 파일 경로 설정

try:
    with open(word_index_path, 'rb') as f:
        char2idx, idx2char = pickle.load(f)
    print(f"'{word_index_path}'에서 char2idx와 idx2char를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print(f"오류: '{word_index_path}' 파일을 찾을 수 없습니다. 해당 파일을 확인해주세요.")
    exit()
except Exception as e:
    print(f"오류: word_index.pkl 파일을 불러오는 중 예기치 않은 오류가 발생했습니다: {e}")
    exit()

# "Juliet: "이라는 문자열을 추가하여 생성된 문장을 result 변수에 저장하세요.
start_seed = "Juliet: "
result = generate_text(model, start_seed)
print(f"\n--- 생성된 텍스트 ---")
print(result)