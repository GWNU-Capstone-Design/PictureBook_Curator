import os
import pickle
import numpy as np
import json
from google.cloud import texttospeech
from flask import Flask, request, jsonify, render_template, redirect, url_for
from openai import OpenAI
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add


#초기 환경 세팅: pytorch 2.5를 사용하기 위해 python 3.9버전 필요
# Initialize Flask app
app = Flask(__name__)
openai_client = OpenAI(api_key="")


# Google 인증 정보 설정
tts_client = texttospeech.TextToSpeechClient.from_service_account_json('./nth-transformer-420210-02d1185d9547.json')


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
max_length = 12  # 최대 캡션 길이(모델 훈련할 때 설정한 값으로 꼭 해야함)

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


def load_json(json_string):
    return json.loads(json_string)

app.jinja_env.filters['load_json'] = load_json


def text_to_speech(text, output_file):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR", 
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    with open(output_file, "wb") as out:
        out.write(response.audio_content)
    return output_file


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

@app.route('/process_caption', methods=['POST'])
def process_caption():
    caption = request.form['caption']
    thread = openai_client.beta.threads.create()
    message = openai_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=caption
    )
    run = openai_client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id='asst_VBfj4zFmZplIZGIGUgJKjUSO',
        instructions="You are given a set of words that summarize all the pages of a storybook and must summarize the content of each page. When given a list of korean words or a theme, you make short story which is delightful and age-appropriate narrative in korean. each story should be less than 20 words. each story should be connected well previous and next summary. You aim to make short story while also subtly teaching lessons or morals. Avoid any content that could be frightening or inappropriate for young children. total summary should be less than 100 words. you should make summary in korean."
    )
    if run.status == 'completed':
        messages = openai_client.beta.threads.messages.list(thread_id=thread.id)
        gpt_result_text = messages.data[0].content[0].text.value
        audio_file_path = text_to_speech(gpt_result_text, 'static/audio/output.mp3')
        return render_template('gpt_result.html', messages=messages, audio_file=audio_file_path)
    else:
        return render_template('error.html', error=run.status)

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
