# 구글 드라이버 연결
from google.colab import drive
drive.mount('/content/drive')

import os

# 'C:/Users/xorua/OneDrive/Desktop/Capston/ImageFiletest/' 경로에 있는 파일 및 디렉토리 목록을 불러옵니다.
train_images_list = os.listdir('C:/Users/xorua/OneDrive/Desktop/Capston/ImageFiletest_Kor/')

# 사용할 샘플의 수를 10으로 제한합니다.
sample_size = 50
train_images_list = train_images_list[:sample_size]

import tensorflow as tf  # 낮은 버전을 실행할 있도록
import cv2
import numpy as np
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
    real_images[j] = np.array(plt.imread('/content/drive/MyDrive/Image/ImageFiletest_Kor2/' + i))
    train[j] = np.array(plt.imread('/content/drive/MyDrive/Image/ImageFiletest_Kor2/' + i))
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
train_captions = pd.read_csv('/content/drive/MyDrive/Image/photo_keywords_Kor2.csv', delimiter='|')

# print(train_captions)

# # 데이터 갯수 출력
# print(len(train_captions))

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

# images_map_caption 함수 호출
captions = images_map_caption(train_images_list, train_captions)

# 반환된 caption 리스트 출력
print(captions)


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
    return tf.Variable(tf.truncated_normal(shape, stddev=0.7), name='W_' + suffix)

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
    layer = tf.nn.conv2d(input=inp, filter=weights, padding='SAME', strides=[1, 1, 1, 1], name='conv_' + suffix)
    layer += biases  # 편향을 합성곱 결과에 추가
    layer = tf.nn.relu6(layer, name='relu_' + suffix)  # ReLU 활성화 함수 적용
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
    layer_shape = layer.get_shape()  # 레이어의 모양(shape)을 가져옵니다.
    num_features = layer_shape[1:4].num_elements()  # 레이어의 특성(feature) 개수를 계산합니다.
    layer = tf.reshape(layer, [-1, num_features], name='flat_' + suffix)  # 평탄화 작업을 수행하여 2D 텐서로 변환합니다.
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
    weights = create_weights([num_inputs, num_outputs], suffix)  # 입력과 출력 사이의 가중치 생성
    biases = create_biases(num_outputs, suffix)  # 출력에 대한 편향 생성
    layer = tf.matmul(inp, weights) + biases  # 가중치와 입력의 행렬 곱셈 후 편향을 더하여 출력 계산
    if use_relu:  # ReLU 활성화 함수 사용 여부 확인
        layer = tf.nn.relu(layer)  # ReLU 활성화 함수를 적용하여 비선형성 추가
    return layer  # 생성된 완전 연결 레이어 반환


def rnn_cell(Win ,Wout, Wfwd, b, hprev, inp):
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
    # 입력과 이전 상태의 출력을 이용하여 현재 상태의 출력 계산
    h = tf.tanh(tf.add(tf.add(tf.matmul(inp, Win), tf.matmul(hprev, Wfwd)), b))
    # 현재 상태의 출력을 이용하여 최종 출력 계산
    out = tf.matmul(h, Wo)
    return h, out  # 현재 상태의 출력과 최종 출력 반환

import tensorflow as tf

tf.device("/device:GPU:0")

learning_rate = 0.001  # 학습률 (learning rate): 가중치 업데이트에 사용되는 스케일 파라미터로, 학습 속도를 조절합니다.

training_iters = 100  # 학습 반복 횟수 (training iterations): 전체 학습 데이터셋에 대해 반복하는 횟수로, 모델이 학습하는 총 횟수입니다.

display_step = 10  # 결과 출력 간격 (display step): 학습 과정에서 중간 결과를 출력하는 빈도로, 몇 번의 학습 반복 후에 출력할지를 결정합니다.

max_sent_limit = 50  # 최대 문장 길이 (maximum sentence limit): 입력 문장의 최대 길이로, 이보다 긴 문장은 잘립니다.

num_tests = 12  # 테스트 샘플 수 (number of tests): 테스트에 사용할 샘플의 수로, 학습된 모델을 평가하기 위해 사용됩니다.

bridge_size = 1024  # 브릿지 크기 (bridge size): 두 개의 네트워크 사이에서 정보를 전달하는 브릿지 레이어의 크기로, 중간 표현을 조절하는 역할을 합니다.

keep_prob = 0.3  # 드롭아웃 확률 (dropout probability): 드롭아웃을 적용할 때 각 뉴런이 유지될 확률로, 오버피팅을 방지하기 위해 사용됩니다.

# 수정된 코드  << 이 코드블록을 주석처리하면 안됨
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_caption = tf.placeholder(tf.float32, [None, vocab_size], name='x_caption')
# 캡션 입력 플레이스홀더: 캡션 데이터를 입력으로 받는 플레이스홀더입니다.
# 입력은 one-hot 인코딩된 단어 벡터로 구성되며, None은 배치 크기를 의미합니다.
# vocab_size는 단어 집합의 크기를 나타냅니다.

x_inp = tf.placeholder(tf.float32, shape=[1, size[0], size[1], num_channels], name='x_image')
# 이미지 입력 플레이스홀더: 이미지 데이터를 입력으로 받는 플레이스홀더입니다.
# 입력은 이미지의 픽셀 값으로 구성된 4차원 텐서로, [배치 크기, 이미지 높이, 이미지 너비, 채널 수] 형태를 가집니다.
# 여기서 배치 크기는 1로 고정되어 있습니다. size는 이미지의 크기를 나타내는 튜플이며, num_channels는 채널 수를 나타냅니다.

y = tf.placeholder(tf.float32, [None, vocab_size], name='y_caption')
# 출력 플레이스홀더: 출력 데이터를 입력으로 받는 플레이스홀더입니다.
# 입력은 캡션 데이터의 실제 값(one-hot 인코딩된 단어 벡터)으로 구성되며, None은 배치 크기를 의미합니다.
# vocab_size는 단어 집합의 크기를 나타냅니다.

Wconv = tf.Variable(tf.truncated_normal([bridge_size, vocab_size], stddev=0.7))
# 컨볼루션 가중치: 브릿지 레이어와 출력 어휘 사이의 가중치 행렬로, 정보 전달을 담당합니다.
# 브릿지 크기 × 어휘 크기의 행렬로 초기화되며, 표준편차 0.7인 절단정규분포에서 랜덤하게 선택된 값으로 초기화됩니다.

bconv = tf.Variable(tf.zeros([1, vocab_size]))
# 컨볼루션 편향: 출력 어휘의 크기만큼의 요소를 가지는 편향 벡터로, 각 출력 단어에 추가적인 편향을 제공합니다.
# 1 × 어휘 크기의 크기로 초기화되며, 모든 요소가 0으로 초기화됩니다.

Wi = tf.Variable(tf.truncated_normal([vocab_size, vocab_size], stddev=0.7))
# 입력 가중치: 입력 단어와 다음 단어 사이의 가중치 행렬로, RNN 셀에 입력으로 제공됩니다.
# 어휘 크기 × 어휘 크기의 행렬로 초기화되며, 표준편차 0.7인 절단정규분포에서 랜덤하게 선택된 값으로 초기화됩니다.

Wf = tf.Variable(tf.truncated_normal([vocab_size, vocab_size], stddev=0.7))
# 순환 가중치: 이전 상태의 출력과 다음 상태의 출력 사이의 가중치 행렬로, RNN 셀의 순환 동작을 담당합니다.
# 어휘 크기 × 어휘 크기의 행렬로 초기화되며, 표준편차 0.7인 절단정규분포에서 랜덤하게 선택된 값으로 초기화됩니다.

Wo = tf.Variable(tf.truncated_normal([vocab_size, vocab_size], stddev=0.7))
# 출력 가중치: RNN 셀의 출력과 다음 단어의 출력 사이의 가중치 행렬로, 최종 출력을 생성합니다.
# 어휘 크기 × 어휘 크기의 행렬로 초기화되며, 표준편차 0.7인 절단정규분포에서 랜덤하게 선택된 값으로 초기화됩니다.

b = tf.Variable(tf.zeros([1, vocab_size]))
# 순환 편향: RNN 셀의 출력에 추가적인 편향을 제공하는 편향 벡터입니다.
# 1 × 어휘 크기의 크기로 초기화되며, 모든 요소가 0으로 초기화됩니다.


layer_conv1 = conv_layer(inp=x_inp, kernel_shape=(3, 3), num_kernels=32, num_channels=3, suffix='1')
# 첫 번째 합성곱 레이어: 입력 이미지에 3x3 크기의 필터를 적용하여 32개의 특징 맵을 생성합니다.
# 입력 이미지의 채널 수는 3으로 가정합니다. 생성된 특징 맵은 렐루 활성화 함수를 통과됩니다.

layer_conv2 = conv_layer(inp=layer_conv1, kernel_shape=(3, 3), num_kernels=32, num_channels=32, suffix='2')
# 두 번째 합성곱 레이어: 첫 번째 레이어의 출력에 3x3 크기의 필터를 적용하여 32개의 특징 맵을 생성합니다.
# 이 레이어에서도 렐루 활성화 함수를 적용합니다.

maxpool1 = tf.nn.max_pool(layer_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2,1], padding= 'SAME')
# 최대 풀링 레이어: 두 번째 합성곱 레이어의 출력에 최대 풀링을 적용하여 공간 차원을 줄입니다.
# ksize는 최대 풀링 윈도우의 크기이며, strides는 스트라이드입니다.

layer_conv3 = conv_layer(inp=maxpool1, kernel_shape=(3, 3), num_kernels=64, num_channels=32, suffix='3')
# 세 번째 합성곱 레이어: 최대 풀링 레이어의 출력에 3x3 크기의 필터를 적용하여 64개의 특징 맵을 생성합니다.
# 이 레이어에서도 렐루 활성화 함수를 적용합니다.

layer_conv4 = conv_layer(inp=layer_conv3, kernel_shape=(3, 3), num_kernels=64, num_channels=64, suffix='4')
# 네 번째 합성곱 레이어: 세 번째 레이어의 출력에 3x3 크기의 필터를 적용하여 64개의 특징 맵을 생성합니다.
# 이 레이어에서도 렐루 활성화 함수를 적용합니다.

maxpool2 = tf.nn.max_pool(layer_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2,1], padding= 'SAME')
# 최대 풀링 레이어: 네 번째 합성곱 레이어의 출력에 최대 풀링을 적용하여 공간 차원을 줄입니다.

layer_conv5 = conv_layer(inp=maxpool2, kernel_shape=(3, 3), num_kernels=128, num_channels=64, suffix='5')
# 다섯 번째 합성곱 레이어: 최대 풀링 레이어의 출력에 3x3 크기의 필터를 적용하여 128개의 특징 맵을 생성합니다.
# 이 레이어에서도 렐루 활성화 함수를 적용합니다.

layer_conv6 = conv_layer(inp=layer_conv5, kernel_shape=(3, 3), num_kernels=128, num_channels=128, suffix='6')
# 여섯 번째 합성곱 레이어: 다섯 번째 레이어의 출력에 3x3 크기의 필터를 적용하여 128개의 특징 맵을 생성합니다.
# 이 레이어에서도 렐루 활성화 함수를 적용합니다.

maxpool3 = tf.nn.max_pool(layer_conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2,1], padding= 'SAME')
# 최대 풀링 레이어: 여섯 번째 합성곱 레이어의 출력에 최대 풀링을 적용하여 공간 차원을 줄입니다.

layer_conv7 = conv_layer(inp=maxpool3, kernel_shape=(3, 3), num_kernels=256, num_channels=128, suffix='7')
# 일곱 번째 합성곱 레이어: 최대 풀링 레이어의 출력에 3x3 크기의 필터를 적용하여 256개의 특징 맵을 생성합니다.
# 이 레이어에서도 렐루 활성화 함수를 적용합니다.

layer_conv8 = conv_layer(inp=layer_conv7, kernel_shape=(3, 3), num_kernels=256, num_channels=256, suffix='8')
# 여덟 번째 합성곱 레이어: 일곱 번째 레이어의 출력에 3x3 크기의 필터를 적용하여 256개의 특징 맵을 생성합니다.
# 이 레이어에서도 렐루 활성화 함수를 적용합니다.

flat_layer = flatten_layer(layer_conv8, suffix='9')
#flat_layer = tf.layers.dropout(flat_layer, rate= keep_prob)
# 플래튼 레이어: 여덟 번째 합성곱 레이어의 출력을 1차원으로 평탄화합니다.

dense_layer_1 = dense_layer(inp=flat_layer, num_inputs=262144, num_outputs=bridge_size, suffix='10')
# 첫 번째 밀집 레이어: 평탄화된 특징 맵을 입력으로 받아, 입력 크기가 262144이고 출력 크기가 bridge_size인 밀집 레이어를 정의합니다.
# 이 레이어에서는 주어진 입력에 대한 선형 변환을 수행한 후, 렐루 활성화 함수를 적용합니다.

start_hook = tf.cast(csr_matrix(([1], ([0], [fwd_dict[start_tag]])), shape=(1, vocab_size)).A, tf.float32)
# 시작 태그에 해당하는 단어를 표현하는 희소 행렬을 생성합니다.
# start_tag를 fwd_dict를 통해 해당하는 인덱스로 변환한 뒤, 해당 인덱스에는 1의 값을 가지고 나머지는 0으로 채웁니다.
# 이를 CSR(Compressed Sparse Row) 형식으로 표현한 후, TensorFlow의 float32 데이터 타입으로 캐스팅합니다.

end_hook = tf.cast(csr_matrix(([1], ([0], [fwd_dict[end_tag]])), shape=(1, vocab_size)).A, tf.float32)
# 종료 태그에 해당하는 단어를 표현하는 희소 행렬을 생성합니다.
# end_tag를 fwd_dict를 통해 해당하는 인덱스로 변환한 뒤, 해당 인덱스에는 1의 값을 가지고 나머지는 0으로 채웁니다.
# 이를 CSR(Compressed Sparse Row) 형식으로 표현한 후, TensorFlow의 float32 데이터 타입으로 캐스팅합니다.

hook = tf.slice(x_caption, [0, 0], [1, vocab_size])
# 입력 캡션 데이터에서 첫 번째 배치의 캡션 벡터를 추출합니다.
# 첫 번째 차원의 크기는 배치 크기를 의미하며, 두 번째 차원의 크기는 단어 집합의 크기인 vocab_size입니다.

h = dense_layer_1
# 이전의 밀집 레이어(dense_layer_1)의 출력을 현재의 은닉 상태로 사용합니다.
# 이는 이미지 정보와 캡션 정보를 조합하기 위한 RNN 셀의 입력으로 활용됩니다.

h, out = rnn_cell(Wi, Wo, Wconv, bconv, h, hook)
# RNN 셀에 입력으로 사용될 이전의 은닉 상태(h)와 현재의 입력(hook)을 전달하여
# 다음 시간 단계에서의 은닉 상태(h)와 출력(out)을 계산합니다.
# Wi와 Wo는 입력과 은닉 상태의 가중치, Wconv는 이미지와 캡션 조합의 가중치를 나타냅니다.
# bconv는 편향(bias)을 나타냅니다.

def fn(prev, curr):
    h = prev[0]
    # 이전 시간 단계의 은닉 상태(prev)에서 첫 번째 요소를 가져와 현재 시간 단계의 은닉 상태(h)로 설정합니다.

    curr = tf.reshape(curr, [1, vocab_size])
    # 현재 시간 단계의 입력(curr)을 1차원의 텐서로 변환합니다.
    # 이는 RNN 셀에 현재 입력을 전달하기 위한 준비 작업입니다.

    h, out = rnn_cell(Wi, Wo, Wf, b, h, curr)
    # RNN 셀에 이전 시간 단계의 은닉 상태(h)와 현재 시간 단계의 입력(curr)을 전달하여
    # 다음 시간 단계에서의 은닉 상태(h)와 출력(out)을 계산합니다.
    # Wi와 Wo는 입력과 은닉 상태의 가중치, Wf는 현재 입력과 은닉 상태의 가중치를 나타냅니다.
    # b는 편향(bias)을 나타냅니다.

    return h, out

_, output = tf.scan(fn, x_caption[1:], initializer=(h, out))
# tf.scan 함수를 사용하여 반복적으로 RNN 셀(fn)을 적용합니다.
# 입력으로 x_caption[1:]을 사용하여 캡션 데이터의 첫 번째 요소를 제외하고 전달합니다.
# 초기값으로는 이전 시간 단계의 은닉 상태(h)와 출력(out)을 설정합니다.
# tf.scan 함수는 각 시간 단계마다 RNN 셀을 적용하고, 각 시간 단계의 출력을 output에 저장합니다.

output = tf.squeeze(output, axis=1)
# 출력 텐서(output)의 차원 중 사이즈가 1인 차원을 제거하여 차원을 축소합니다.
# 이는 RNN 셀의 출력을 1차원으로 변환하여 다음 계층으로 전달하기 위한 작업입니다.

outputs = tf.concat([out, output], axis=0)
# RNN 셀의 출력(out)과 변환된 출력(output)을 수직으로 연결(concatenate)하여 하나의 텐서(outputs)로 합칩니다.
# 이는 RNN 셀의 출력과 변환된 출력을 결합하여 최종 출력을 생성하기 위한 작업입니다.

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=y))
# Softmax 크로스 엔트로피 손실 함수를 사용하여 모델의 예측값(outputs)과 실제값(y) 사이의 손실을 계산합니다.
# 이 손실 값은 모델의 성능을 평가하는 데 사용됩니다.

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# Adam 옵티마이저를 사용하여 손실 함수를 최소화하는 학습 과정을 정의합니다.
# 이를 위해 학습률(learning_rate)을 설정하여 모델의 가중치를 업데이트합니다.

pred = tf.nn.softmax(outputs)
# 출력 텐서(outputs)에 소프트맥스 함수를 적용하여 각 클래스(단어)에 대한 확률을 계산합니다.
# 이를 통해 모델의 출력을 확률 분포로 변환합니다.

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# 모델의 예측값(pred)과 실제값(y)을 비교하여 각 샘플에 대한 정확한 예측 여부를 판별합니다.
# tf.equal 함수를 사용하여 예측값과 실제값이 일치하는지 여부를 확인합니다.

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# 정확도(accuracy)를 계산합니다.
# 각 샘플에 대한 정확한 예측 여부를 0 또는 1로 캐스팅한 후, 이를 평균하여 전체 데이터셋에 대한 정확도를 계산합니다.

out_tensor = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size = 0)

out_tensor = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
# 출력을 저장할 TensorArray를 생성합니다.
# dtype은 출력의 데이터 타입을 지정하고, dynamic_size=True는 TensorArray의 크기를 동적으로 조절할 수 있도록 합니다.
# size=0은 초기 크기를 0으로 설정합니다.

htest = dense_layer_1
# 테스트용 은닉 상태(htest)를 초기화합니다.
# 이는 RNN 셀의 초기 은닉 상태로 사용됩니다.

htest, out_first = rnn_cell(Wi, Wo, Wconv, bconv, htest, start_hook)
# RNN 셀에 초기 은닉 상태(htest)와 시작 토큰(start_hook)을 입력으로 전달하여 첫 번째 시간 단계의 은닉 상태와 출력을 계산합니다.
# 이는 모델의 초기 상태를 설정하는 단계입니다.

t = 0
# 시간 스텝을 나타내는 변수 t를 초기화합니다.

out_ = tf.one_hot(tf.argmax(tf.nn.softmax(out_first), 1), depth=vocab_size)
# softmax 함수를 사용하여 출력(out_first)을 확률 분포로 변환하고, argmax 함수를 사용하여 가장 높은 확률을 가진 단어를 선택합니다.
# 선택된 단어를 one-hot 인코딩하여 out_에 저장합니다.

out_tensor = out_tensor.write(t, out_)
# 현재 시간 단계의 출력(out_)을 TensorArray에 저장합니다.

t += 1
# 시간 스텝 변수 t를 증가시킵니다.

def condition(res, h, out_tensor, t):
    # 반복을 계속할 조건을 정의하는 함수입니다.
    # res: 현재 시간 단계의 출력
    # h: 현재 시간 단계의 은닉 상태
    # out_tensor: 출력을 저장하는 TensorArray
    # t: 현재 시간 단계

    # 현재 시간 단계의 출력(res)을 argmax 함수를 사용하여 가장 높은 확률을 가진 단어의 인덱스로 변환하고,
    # 이를 end_tag의 인덱스와 비교하여 end_tag가 아닌지 확인합니다.
    # 또한, 시간 스텝 변수 t가 최대 문장 길이(max_sent_limit)보다 작은지 확인합니다.
    return tf.logical_and(tf.logical_not(tf.equal(tf.argmax(res, 1)[0], fwd_dict[end_tag])), tf.less(t, max_sent_limit))

def action(res, h, out_tensor, t):
    # 각 반복 단계에서 수행할 작업을 정의하는 함수입니다.
    # res: 현재 시간 단계의 출력
    # h: 현재 시간 단계의 은닉 상태
    # out_tensor: 출력을 저장하는 TensorArray
    # t: 현재 시간 단계

    # RNN 셀에 현재 시간 단계의 출력(res)과 은닉 상태(h)를 입력으로 전달하여 다음 시간 단계의 은닉 상태와 출력을 계산합니다.
    h, out = rnn_cell(Wi, Wo, Wf, b, h, res)

    # softmax 함수를 사용하여 출력(out)을 확률 분포로 변환하고, argmax 함수를 사용하여 가장 높은 확률을 가진 단어의 인덱스로 변환합니다.
    res = tf.one_hot(tf.argmax(tf.nn.softmax(out), 1), depth=vocab_size)

    # 현재 시간 단계의 출력(res)을 TensorArray에 저장합니다.
    out_tensor = out_tensor.write(t, res)

    # 시간 스텝 변수 t를 1 증가시킵니다.
    return res, h, out_tensor, t + 1
_, __, final_outputs, T = tf.while_loop(condition, action, [out_, htest, out_tensor, t])
# while_loop 함수를 사용하여 반복적으로 action 함수를 호출하여 모델의 출력을 생성합니다.
# condition 함수가 False를 반환하거나 최대 문장 길이에 도달할 때까지 반복됩니다.
# 초기 상태로는 시작 토큰에 대한 출력(out_), 테스트용 은닉 상태(htest), 출력을 저장하는 TensorArray(out_tensor), 시간 스텝 변수 t를 사용합니다.

final_prediction = tf.squeeze(final_outputs.stack())
# TensorArray에 저장된 모든 출력을 하나의 텐서로 병합합니다.
# 이는 모델의 최종 출력을 나타냅니다.

saver = tf.train.Saver()
# 모델의 변수를 저장하고 복원하기 위한 Saver 객체를 생성합니다.

init = tf.global_variables_initializer()
# 변수들을 초기화하는 연산을 생성합니다.

with tf.Session() as sess:
    # 변수 초기화
    sess.run(init)

    # 훈련 샘플의 개수 구하기
    m = len(train_caption)

    # 에포크를 반복하여 훈련
    for epoch in range(training_iters):
        total_cost = 0
        total_acc = 0

        # 각 훈련 예제에 대해 반복
        for i in range(m):
            # 옵티마이저, 비용, 정확도 연산 실행
            _, cst, acc = sess.run([optimizer, cost, accuracy], feed_dict = {x_caption:train_caption[i][:-1].A, x_inp:train[i:i+1], y:train_caption[i][1:].A})

            # 총 비용과 정확도 누적
            total_cost += cst
            total_acc += acc

        # 'display_step' 에포크마다 진행 상황 표시
        if (epoch + 1) % display_step == 0:
            print('After ', (epoch + 1), 'iterations: Cost = ', total_cost / m, 'and Accuracy: ', total_acc * 100/ m , '%' )

        if total_acc * 100/ m >= 95:
          break
    saver = tf.train.Saver()
    saver.save(sess, "/content/drive/MyDrive/model/test")

    # 훈련 완료
    print('최적화 완료!')
    print("테스트 시작")

    # 훈련된 모델 테스트
    for tests in range(num_tests):
        # 훈련 세트에서 이미지를 무작위로 선택
        image_num = random.randint(0, sample_size - 1)

        # 선택한 이미지에 대한 예측된 캡션 가져오기
        caption = sess.run(final_prediction, feed_dict = {x_inp:train[image_num:image_num + 1]})

        # 예측된 캡션의 형태 출력 (디버깅용)
        print(caption.shape)

        # 예측된 캡션을 단어로 변환하여 출력
        caption = np.argmax(caption[:-1], 1)
        capt = ''
        for i in caption:
            capt += rev_dict[i] + ' '
        print('예측된 캡션:->', capt)

        # 선택한 이미지의 원래 캡션 가져오기
        orig_cap = np.argmax(train_caption[image_num:image_num + 1][0][1:-1].A, 1)
        orignalcaption = ''
        for i in orig_cap:
            orignalcaption += rev_dict[i] + ' '
        print('원본 캡션:->', orignalcaption)

        # 선택한 이미지 표시
        plt.imshow(real_images[image_num])
        plt.title('Image')
        plt.show()
