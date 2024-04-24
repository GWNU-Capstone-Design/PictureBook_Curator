function synthesizeText() {
  const text = document.getElementById('textInput').value;
  const genderValue = document.querySelector('input[name="gender"]:checked').value;
  const gender = genderValue === 'MALE' ? 'MALE' : 'FEMALE';

  fetch('/synthesize', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text, gender }),
  })
    .then(response => response.json())
    .then(data => {
      const audioElement = document.getElementById('audioPlayer');
      audioElement.src = `data:audio/mp3;base64,${data.audioContent}`;
      audioElement.play();
    })
    .catch(error => console.error('Error:', error));
}