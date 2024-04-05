const bookContainer = document.querySelector('.main-content');

// 예시 이미지 파일 경로
const imagePaths = ['test.jpg', 'test.jpg', 'test.jpg'];

// 이미지 파일을 book 클래스에 추가
imagePaths.forEach(path => {
  const bookDiv = document.createElement('div');
  bookDiv.classList.add('book');

  const img = document.createElement('img');
  img.src = path;

  bookDiv.appendChild(img);
  bookContainer.appendChild(bookDiv);
});