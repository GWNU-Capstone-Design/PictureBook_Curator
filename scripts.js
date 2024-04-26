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

//책 추가부분
const addBookBtn = document.querySelector('.add-book');
const modal = document.getElementById('addBookModal');
const closeModal = document.querySelector('.close');

addBookBtn.addEventListener('click', function() {
  modal.style.display = "block";
});

closeModal.addEventListener('click', function() {
  modal.style.display = "none";
});

window.addEventListener('click', function(event) {
  if (event.target == modal) {
    modal.style.display = "none";
  }
});

function previewCover(files) {
  const preview = document.getElementById('coverPreview');
  preview.innerHTML = ''; // 기존 미리보기를 지웁니다.
  if (files[0]) {
    const reader = new FileReader();
    reader.onload = function(e) {
      const img = document.createElement('img');
      img.src = e.target.result;
      preview.appendChild(img);
    };
    reader.readAsDataURL(files[0]);
  }
}

function previewContent(files) {
  const preview = document.getElementById('contentPreview');
  preview.innerHTML = '';
  Array.from(files).forEach(file => {
    const reader = new FileReader();
    reader.onload = function(e) {
      const img = document.createElement('img');
      img.src = e.target.result;
      preview.appendChild(img);
    };
    reader.readAsDataURL(file);
  });
}

document.getElementById('bookForm').addEventListener('submit', function(event) {
  event.preventDefault();
  alert('책이 추가되었습니다!');
  modal.style.display = "none";
});

