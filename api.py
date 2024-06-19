import os
import pickle
import numpy as np
from google.cloud import texttospeech
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from openai import OpenAI
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__, static_folder='static')

CORS(app, resources={r"/static/audio/*": {"origins": "*"}})  # CORS 설정 추가
#오디오 사용을 위해 라우트 추가
@app.route('/static/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory('static/audio', filename)

openai_client = OpenAI(api_key="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")

# Google authentication settings
tts_client = texttospeech.TextToSpeechClient.from_service_account_json('./nth-transformer-420210-02d1185d9547.json')

# Directory settings
UPLOAD_FOLDER = os.path.join('static', 'uploads')
AUDIO_FOLDER = os.path.join('static', 'audio')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

WORKING_DIR = ""
FEATURES_PATH = os.path.join(WORKING_DIR, 'features.pkl')
MODEL_PATH = os.path.join(WORKING_DIR, 'model.h5')
TOKENIZER_PATH = os.path.join(WORKING_DIR, 'tokenizer.pkl')

# Load VGG16 model and modify output layer
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Load features
with open(FEATURES_PATH, 'rb') as f:
    features = pickle.load(f)

# Load tokenizer
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
max_length = 12  # Maximum caption length (set during model training)

# Load trained model
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

@app.route('/process_images', methods=['POST'])
def process_images():
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images uploaded'}), 400

        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': 'No selected files'}), 400

        captions = []
        image_urls = []
        gpt_audio_urls = []
        for file in files:
            if file:
                img_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(img_path)

                # Extract features from image and generate caption
                feature = extract_features(img_path)
                caption = generate_caption(feature)
                captions.append(caption)
                image_urls.append(img_path)

        # Process captions with GPT
        combined_captions = '\n'.join([f"{i+1}번: {caption}" for i, caption in enumerate(captions)])
        thread = openai_client.beta.threads.create()
        message = openai_client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=combined_captions
        )
        run = openai_client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id='asst_VBfj4zFmZplIZGIGUgJKjUSO',
            instructions="You are given a set of words that summarize all the pages of a storybook and must summarize the content of each page. When given a list of korean words or a theme, you make short story which is delightful and age-appropriate narrative in korean. each story should be less than 20 words. each story should be connected well previous and next summary. You aim to make short story while also subtly teaching lessons or morals. Avoid any content that could be frightening or inappropriate for young children. total summary should be less than 100 words. you should make summary in korean."
        )
        if run.status == 'completed':
            messages = openai_client.beta.threads.messages.list(thread_id=thread.id)
            gpt_result_text = messages.data[0].content[0].text.value
            gpt_results = gpt_result_text.split('\n')

            for i, gpt_result in enumerate(gpt_results):
                clean_gpt_result = gpt_result.split(':', 1)[1].strip()  # 'n번:' 제거
                image_name = os.path.splitext(os.path.basename(image_urls[i]))[0]
                audio_file_path = os.path.join(AUDIO_FOLDER, f'{image_name}.mp3')
                text_to_speech(clean_gpt_result, audio_file_path)
                gpt_audio_urls.append(audio_file_path)
            
            # Include text info and audio file paths in JSON response
            response_data = {
                "image_urls": image_urls,
                "captions": captions,
                "gpt_results": gpt_results,
                "gpt_audio_urls": gpt_audio_urls
            }
            return jsonify(response_data)
        else:
            return jsonify({'error': run.status}), 500
    except Exception as e:
        print(f"Error processing images: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
