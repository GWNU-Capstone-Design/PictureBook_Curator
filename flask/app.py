import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

app = Flask(__name__)

# 경로 설정
BASE_DIR = "image"
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

WORKING_DIR = ""
FEATURES_PATH = os.path.join(WORKING_DIR, 'features.pkl')
MODEL_PATH = os.path.join(WORKING_DIR, 'model.h5')
TOKENIZER_PATH = os.path.join(WORKING_DIR, 'tokenizer.pkl')

# VGG16 모델 로드 및 출력 레이어 변경
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# 특징 로드
with open(FEATURES_PATH, 'rb') as f:
    features = pickle.load(f)

# 토크나이저 로드
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
max_length = 10  # 최대 캡션 길이(모델 훈련할 때 설정한 값으로 꼭 해야함)

# 학습된 모델 로드
model = load_model(MODEL_PATH)

def extract_features(img_path):
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    return feature

def generate_caption(feature):
    input_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([input_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = None
        for word_, index in tokenizer.word_index.items():
            if index == yhat:
                word = word_
                break
        if word is None:
            break
        input_text += ' ' + word
        if word == 'endseq':
            break
    return input_text.replace('startseq', '').replace('endseq', '').strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    if file:
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        feature = extract_features(img_path)
        caption = generate_caption(feature)

        return render_template('index.html', caption=caption, image_path=url_for('static', filename='uploads/' + file.filename))

@app.route('/caption', methods=['POST'])
def caption_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    img = request.files['image']
    img_path = os.path.join(UPLOAD_FOLDER, img.filename)
    img.save(img_path)

    feature = extract_features(img_path)
    caption = generate_caption(feature)
    return jsonify({'caption': caption})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
