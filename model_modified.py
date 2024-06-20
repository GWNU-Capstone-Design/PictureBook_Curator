from keras.models import Model
from keras.layers import Input, Dropout, Dense, Embedding, LSTM, add
from keras.utils import plot_model, to_categorical, pad_sequences
import keras.backend as K
import tensorflow as tf
import numpy as np

# 모델 정의
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)


# 커스텀 Loss 함수 정의
def custom_loss(y_true, y_pred):
    loss = K.categorical_crossentropy(y_true, y_pred)

    # 중복 단어 패널티 추가
    y_pred_ids = K.argmax(y_pred, axis=-1)
    unique, _, counts = tf.unique_with_counts(y_pred_ids)
    counts = K.cast(counts, K.floatx())
    penalty = K.sum(K.relu(counts - 1))

    return loss + penalty


# 모델 컴파일
model.compile(loss=custom_loss, optimizer='adam')

# 모델 구조 보여주기
plot_model(model, show_shapes=True)


