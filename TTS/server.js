const express = require('express');
const bodyParser = require('body-parser');
const textToSpeech = require('@google-cloud/text-to-speech');

const app = express();
const port = 3000;

// Google TTS 클라이언트 설정
const client = new textToSpeech.TextToSpeechClient({
  keyFilename: '/nth-transformer-420210-02d1185d9547.json' // 인증 정보 파일 경로
});

app.use(bodyParser.json());
app.use(express.static('public')); // 정적 파일 제공 폴더 설정

// TTS 처리 라우트
app.post('/synthesize', async (req, res) => {
  const text = req.body.text;
  const request = {
    input: { text },
    voice: { languageCode: 'en-US', ssmlGender: 'NEUTRAL' },
    audioConfig: { audioEncoding: 'MP3' },
  };

  try {
    const [response] = await client.synthesizeSpeech(request);
    const audioContent = response.audioContent;
    res.send({ audioContent: audioContent.toString('base64') });
  } catch (error) {
    console.error('Error:', error);
    res.status(500).send('Failed to synthesize speech.');
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}/`);
});
