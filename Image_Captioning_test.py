import os

# 'C:/Users/xorua/OneDrive/Desktop/Capston/ImageFiletest/' 경로에 있는 파일 및 디렉토리 목록을 불러옵니다.
train_images_list = os.listdir('C:/Users/xorua/OneDrive/Desktop/Capston/ImageFiletest_Kor/')

# 사용할 샘플의 수를 10으로 제한합니다.
sample_size = 50
train_images_list = train_images_list[:sample_size]

import tensorflow as tf
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import random

# 사용할 이미지의 크기 (256 x 256) 및 채널 수 (RGB이므로 3)를 설정합니다.
size = (256, 256)
num_channels = 3

# 훈련 데이터와 실제 이미지 데이터를 저장할 numpy 배열을 초기화합니다. 크기는 sample_size로 설정합니다.
train = np.array([None] * sample_size)
real_images = np.array([None] * sample_size)

# 지정된 경로에서 이미지를 불러와서 `real_images`와 `train` 배열에 저장합니다.
j = 0
for i in train_images_list:
    real_images[j] = np.array(plt.imread('C:/Users/xorua/OneDrive/Desktop/Capston/ImageFiletest_Kor/' + i))
    train[j] = np.array(plt.imread('C:/Users/xorua/OneDrive/Desktop/Capston/ImageFiletest_Kor/' + i))
    j += 1

# `train` 배열에 저장된 모든 이미지를 지정된 크기로 변경하고 적절한 차원으로 재구성합니다.
j = 0
for i in train:
    # 이미지를 지정된 크기(size)로 조정합니다.
    train[j] = cv2.resize(i, size)
    # 이미지를 모델 입력에 맞는 형태로 재구성합니다. 여기서는 (1, 256, 256, 3) 형태로 재구성합니다.
    train[j] = train[j].reshape(1, size[0], size[1], num_channels)
    j += 1

# train 배열의 모든 원소를 수직으로 쌓아 하나의 numpy 배열로 합칩니다.
train = np.vstack(train[:])

# # 첫 번째 훈련 이미지를 보여줍니다.
# plt.imshow(np.squeeze(train[0]))
# plt.show()

import pandas as pd
# csv 파일 읽기
train_captions = pd.read_csv('C:/Users/xorua/OneDrive/Desktop/Capston/csvFile/photo_keywords_Kor.csv',  delimiter='|')

# 파일 구문자 구별 및 파일 확장자 제거
def get_images_id(names):
    names = [int(x.split('_')[-1].split('.')[0]) for x in names]
    return names
# print(train_captions)

# ids = get_images_id(train_images_list[:sample_size])
train_captions.columns = ['image_name', 'comment_number', 'comment']

def images_map_caption(train_images_list, train_captions):
    """
    이미지 파일 이름과 해당 이미지에 대한 캡션을 포함하는 데이터프레임에서 각 이미지에 대한 캡션을 추출하여 리스트로 반환하는 함수.

    Parameters:
        train_images_list (list): 이미지 파일 이름을 포함하는 리스트.
        train_captions (DataFrame): 이미지 파일 이름과 해당 이미지에 대한 캡션을 포함하는 데이터프레임.

    Returns:
        list: 각 이미지에 대한 캡션을 저장한 리스트.

    """
    caption = []
    for i in train_images_list:
        caption.append(train_captions[train_captions['image_name'] == i]['comment'].iat[0])
    return caption

# 이미지 파일 이름과 해당 이미지에 대한 캡션을 포함하는 데이터프레임에서 각 이미지에 대한 캡션을 추출하여 NumPy 배열로 변환하는 코드입니다.

# train_images_list와 train_captions를 입력으로 사용하여 images_map_caption() 함수를 호출하여 각 이미지에 대한 캡션을 추출합니다.
# 반환된 캡션들을 NumPy 배열로 변환하여 captions 변수에 저장합니다.
captions = np.array(images_map_caption(train_images_list, train_captions))

# print(captions)

# captions 배열의 모양(shape)을 출력하여 배열의 차원과 각 차원의 크기를 확인합니다.
print(captions.shape)

import re

# 시작 태그와 끝 태그 지정
start_tag = '<s>'  # 문장의 시작을 나타내는 태그
end_tag = '<e>'    # 문장의 끝을 나타내는 태그

def get_vocab(captions):
    arr = []  # 모든 단어를 저장할 리스트
    m = captions.shape[0]  # captions의 행 개수
    sentence = [None] * m  # captions의 각 문장을 저장할 리스트
    j = 0  # sentence 리스트의 인덱스 변수
    for i in captions:  # captions의 각 문장에 대해 반복
        i = re.sub(' +', ' ', i)  # 중복된 공백을 하나로 줄임
        i = start_tag + ' ' + i + ' ' + end_tag  # 문장에 시작 태그와 끝 태그 추가
        sentence[j] = i.split()  # 문장을 공백을 기준으로 분할하여 리스트로 저장
        j += 1  # 인덱스 변수 증가
        arr = arr + i.split()  # 모든 단어를 저장하는 리스트에 현재 문장의 단어 추가
    arr = list(set(arr))  # 중복을 제거하여 유일한 단어 리스트 생성
    vocab_size = len(arr)  # 유일한 단어의 개수를 저장
    j = 0  # 인덱스 변수 초기화
    fwd_dict = {}  # 단어를 인덱스로 매핑하는 딕셔너리
    rev_dict = {}  # 인덱스를 단어로 매핑하는 딕셔너리
    for i in arr:  # 유일한 단어 리스트의 각 단어에 대해 반복
        fwd_dict[i] = j  # 단어를 인덱스 j로 매핑
        rev_dict[j] = i  # 인덱스 j를 단어로 매핑
        j += 1  # 인덱스 변수 증가
    return vocab_size, sentence, fwd_dict, rev_dict  # 어휘 크기, 각 문장 리스트, 단어-인덱스 딕셔너리, 인덱스-단어 딕셔너리 반환

vocab_size, sentences, fwd_dict, rev_dict = get_vocab(captions)

from scipy.sparse import csr_matrix  # 희소 행렬 생성을 위한 모듈 import
from scipy.sparse import vstack  # 희소 행렬을 수직으로 결합하기 위한 모듈 import
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D
import tensorflow.compat.v1 as tf

# 문장 리스트를 이용하여 학습용 캡션 데이터 생성
m = len(sentences)  # 문장의 개수
train_caption = [None] * m  # 학습용 캡션 데이터를 저장할 리스트 초기화
i = 0  # 리스트의 인덱스 변수 초기화
for sentence in sentences:  # 각 문장에 대해 반복
    cap_array = None  # 문장의 단어를 포함하는 희소 행렬 초기화
    for word in sentence:  # 문장의 각 단어에 대해 반복
        row = [0]  # 행 인덱스
        col = [fwd_dict[word]]  # 열 인덱스: 단어를 인덱스로 변환하여 저장
        data = [1]  # 데이터 값: 단어의 등장 횟수
        if cap_array is None:  # 첫 번째 단어인 경우
            cap_array = csr_matrix((data, (row, col)), shape=(1, vocab_size))  # 희소 행렬 생성
        else:  # 두 번째 이후의 단어인 경우
            cap_array = vstack((cap_array, csr_matrix((data, (row, col)), shape=(1, vocab_size))))  # 희소 행렬을 수직으로 결합
    train_caption[i] = cap_array  # 생성된 캡션 데이터를 리스트에 저장
    i += 1  # 인덱스 변수 증가
    
train_caption[0].shape

# Model Design

# 가중치(Weights)를 생성하는 함수
def create_weights(shape, suffix):
    """
    주어진 모양(shape)의 가중치를 생성하는 함수입니다.

    Parameters:
        shape (tuple): 가중치의 모양을 지정하는 튜플.
        suffix (str): 가중치의 이름에 추가할 접미사.

    Returns:
        tf.Variable: 생성된 가중치 변수.
    """
    # return tf.Variable(tf.truncated_normal(shape, stddev=0.7), name='W_' + suffix)
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.7), name='W_' + suffix)

# 편향(Biases)를 생성하는 함수
def create_biases(size, suffix):
    """
    주어진 크기(size)의 편향을 생성하는 함수입니다.

    Parameters:
        size (int): 편향의 크기.
        suffix (str): 편향의 이름에 추가할 접미사.

    Returns:
        tf.Variable: 생성된 편향 변수.
    """
    return tf.Variable(tf.zeros([size]), name='b_' + suffix)

def conv_layer(inp, kernel_shape, num_channels, num_kernels, suffix):
    """
    합성곱 레이어를 생성하는 함수입니다.

    Parameters:
        inp (tf.Tensor): 입력 텐서.
        kernel_shape (tuple): 커널(필터)의 모양을 지정하는 튜플.
        num_channels (int): 입력 텐서의 채널 수.
        num_kernels (int): 생성될 커널(필터)의 개수.
        suffix (str): 레이어의 이름에 추가할 접미사.

    Returns:
        tf.Tensor: 생성된 합성곱 레이어.
    """
    filter_shape = [kernel_shape[0], kernel_shape[1], num_channels, num_kernels]  # 필터의 모양 지정
    weights = create_weights(shape=filter_shape, suffix=suffix)  # 가중치 생성
    biases = create_biases(num_kernels, suffix=suffix)  # 편향 생성
    # 합성곱 연산 수행
    layer = tf.keras.layers.Conv2D(filters=num_kernels, kernel_size=kernel_shape, padding='SAME', strides=(1, 1), name='conv_' + suffix)(inp)
    layer += biases  # 편향을 합성곱 결과에 추가
    layer = tf.keras.layers.ReLU(name='relu_' + suffix)(layer)  # ReLU 활성화 함수 적용
    return layer  # 생성된 합성곱 레이어 반환


def flatten_layer(layer, suffix):
    """
    주어진 레이어를 평탄화하는 함수입니다.

    Parameters:
        layer (tf.Tensor): 입력 텐서.
        suffix (str): 레이어의 이름에 추가할 접미사.

    Returns:
        tf.Tensor: 평탄화된 레이어.
    """
    layer_shape = layer.shape  # 레이어의 모양(shape)을 가져옵니다.
    num_features = tf.reduce_prod(layer_shape[1:])  # 레이어의 특성(feature) 개수를 계산합니다.
    num_features = num_features.numpy()  # numpy 배열로 변환합니다.
    layer = tf.keras.layers.Reshape((num_features,), name='flat_' + suffix)(layer)  # 평탄화 작업을 수행하여 2D 텐서로 변환합니다.
    return layer  # 변환된 평탄화된 레이어를 반환합니다.

def dense_layer(inp, num_inputs, num_outputs, suffix, use_relu=True):
    """
    주어진 입력으로부터 완전 연결 레이어를 생성하는 함수입니다.

    Parameters:
        inp (tf.Tensor): 입력 텐서.
        num_inputs (int): 입력 특성의 수.
        num_outputs (int): 출력 특성의 수.
        suffix (str): 레이어의 이름에 추가할 접미사.
        use_relu (bool): ReLU 활성화 함수를 사용할지 여부를 지정하는 플래그.

    Returns:
        tf.Tensor: 생성된 완전 연결 레이어.
    """
    # weights = create_weights([num_inputs, num_outputs], suffix)  # 입력과 출력 사이의 가중치 생성
    # biases = create_biases(num_outputs, suffix)  # 출력에 대한 편향 생성
    # layer = tf.matmul(inp, weights) + biases  # 가중치와 입력의 행렬 곱셈 후 편향을 더하여 출력 계산
    # if use_relu:  # ReLU 활성화 함수 사용 여부 확인
    #     layer = tf.nn.relu(layer)  # ReLU 활성화 함수를 적용하여 비선형성 추가
    # return layer  # 생성된 완전 연결 레이어 반환
    dense = tf.keras.layers.Dense(num_outputs, activation='relu' if use_relu else None, name='dense_' + suffix)
    layer = dense(inp)
    return layer

# 오류 발생 지역
def rnn_cell(Win, Wout, Wfwd, b, hprev, inp):
    """
    주어진 입력으로부터 RNN 셀을 생성하는 함수입니다.

    Parameters:
        Win (tf.Tensor): 입력에 대한 가중치 행렬.
        Wout (tf.Tensor): 출력에 대한 가중치 행렬.
        Wfwd (tf.Tensor): 다음 상태에 대한 가중치 행렬.
        b (tf.Tensor): 편향 벡터.
        hprev (tf.Tensor): 이전 상태의 출력.
        inp (tf.Tensor): 현재 입력.

    Returns:
        tuple: 현재 상태의 출력과 출력값.
    """
    # 현재 상태의 출력 계산
    h = tf.tanh(tf.matmul(inp, Win) + tf.matmul(hprev, Wfwd) + b)

    # 최종 출력 계산
    out = tf.matmul(h, Wout)

    return h, out

learning_rate = 0.0001  # 학습률 (learning rate): 가중치 업데이트에 사용되는 스케일 파라미터로, 학습 속도를 조절합니다.

training_iters = 5000  # 학습 반복 횟수 (training iterations): 전체 학습 데이터셋에 대해 반복하는 횟수로, 모델이 학습하는 총 횟수입니다.

display_step = 1000  # 결과 출력 간격 (display step): 학습 과정에서 중간 결과를 출력하는 빈도로, 몇 번의 학습 반복 후에 출력할지를 결정합니다.

max_sent_limit = 50  # 최대 문장 길이 (maximum sentence limit): 입력 문장의 최대 길이로, 이보다 긴 문장은 잘립니다.

num_tests = 12  # 테스트 샘플 수 (number of tests): 테스트에 사용할 샘플의 수로, 학습된 모델을 평가하기 위해 사용됩니다.

bridge_size = 1024  # 브릿지 크기 (bridge size): 두 개의 네트워크 사이에서 정보를 전달하는 브릿지 레이어의 크기로, 중간 표현을 조절하는 역할을 합니다.

keep_prob = 0.3  # 드롭아웃 확률 (dropout probability): 드롭아웃을 적용할 때 각 뉴런이 유지될 확률로, 오버피팅을 방지하기 위해 사용됩니다.


# 이미지 캡션의 입력 데이터를 위한 플레이스홀더 생성
x_caption = tf.keras.Input(shape=(vocab_size,), dtype=tf.float32, name='x_caption')
# 이미지 입력 데이터를 위한 플레이스홀더 생성
# x_inp = tf.keras.Input(shape=(1, size[0], size[1], num_channels), dtype=tf.float32, name='x_image')
x_inp = Input(shape=(size[0], size[1], num_channels), dtype=tf.float32, name='x_image')
# 이미지 캡션의 출력 데이터를 위한 플레이스홀더 생성
y = tf.keras.Input(shape=(vocab_size,), dtype=tf.float32, name='y_caption')


# 가중치 초기화를 위한 표준 편차 설정
stddev = 0.7

# Wconv 가중치 초기화
Wconv = tf.Variable(tf.random.truncated_normal([bridge_size, vocab_size], stddev=stddev))

# bconv 편향 초기화
bconv = tf.Variable(tf.zeros([1, vocab_size]))

# Wi, Wf, Wo 가중치 초기화
Wi = tf.Variable(tf.random.truncated_normal([vocab_size, vocab_size], stddev=stddev))
Wf = tf.Variable(tf.random.truncated_normal([vocab_size, vocab_size], stddev=stddev))
Wo = tf.Variable(tf.random.truncated_normal([vocab_size, vocab_size], stddev=stddev))

# 편향 초기화
b = tf.Variable(tf.zeros([1, vocab_size]))

# 첫 번째 컨볼루션 레이어
layer_conv1 = conv_layer(inp=x_inp, kernel_shape=(3, 3), num_kernels=32, num_channels=3, suffix='1')

# 두 번째 컨볼루션 레이어
layer_conv2 = conv_layer(inp=layer_conv1, kernel_shape=(3, 3), num_kernels=32, num_channels=32, suffix='2')

# 첫 번째 맥스 풀링 레이어
maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer_conv2)

# 세 번째 컨볼루션 레이어
layer_conv3 = conv_layer(inp=maxpool1, kernel_shape=(3, 3), num_kernels=64, num_channels=32, suffix='3')

# 네 번째 컨볼루션 레이어
layer_conv4 = conv_layer(inp=layer_conv3, kernel_shape=(3, 3), num_kernels=64, num_channels=64, suffix='4')

# 두 번째 맥스 풀링 레이어
maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer_conv4)

# 다섯 번째 컨볼루션 레이어
layer_conv5 = conv_layer(inp=maxpool2, kernel_shape=(3, 3), num_kernels=128, num_channels=64, suffix='5')

# 여섯 번째 컨볼루션 레이어
layer_conv6 = conv_layer(inp=layer_conv5, kernel_shape=(3, 3), num_kernels=128, num_channels=128, suffix='6')

# 세 번째 맥스 풀링 레이어
maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer_conv6)

# 일곱 번째 컨볼루션 레이어
layer_conv7 = conv_layer(inp=maxpool3, kernel_shape=(3, 3), num_kernels=256, num_channels=128, suffix='7')

# 여덟 번째 컨볼루션 레이어
layer_conv8 = conv_layer(inp=layer_conv7, kernel_shape=(3, 3), num_kernels=256, num_channels=256, suffix='8')

# 평탄화된 레이어 생성
flat_layer = flatten_layer(layer_conv8, suffix='9')

# 드롭아웃 레이어는 비활성화된 상태로 주석 처리됨
# flat_layer = tf.layers.dropout(flat_layer, rate= keep_prob)

# 완전 연결 레이어 생성
dense_layer_1 = dense_layer(inp=flat_layer, num_inputs=262144 , num_outputs=bridge_size, suffix='10')

# 시작 태그에 해당하는 단어에 대한 one-hot 벡터 생성
start_hook = tf.cast(csr_matrix(([1], ([0], [fwd_dict[start_tag]])), shape=(1, vocab_size)).A, tf.float32)
# 종료 태그에 해당하는 단어에 대한 one-hot 벡터 생성
end_hook = tf.cast(csr_matrix(([1], ([0], [fwd_dict[end_tag]])), shape=(1, vocab_size)).A, tf.float32)


hook = x_caption[:, :vocab_size]
h = dense_layer_1
h, out = rnn_cell(Wi, Wo, Wconv, bconv, h, hook)