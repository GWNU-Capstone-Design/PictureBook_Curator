// 책 추가 버튼과 모달 동작
const addBookBtn = document.querySelector('.add-book');
const bookModal = document.getElementById('addBookModal');
const contentModal = document.getElementById('contentUploadModal');
const closeButtons = document.querySelectorAll('.close');

// 모달 열기
addBookBtn.addEventListener('click', function() {
    bookModal.style.display = "block";
});

// 모든 모달 닫기 버튼
closeButtons.forEach(button => {
    button.addEventListener('click', function() {
        this.parentElement.parentElement.parentElement.style.display = "none";
    });
});

// 클릭 시 모달 밖을 클릭하면 모달 닫기
window.addEventListener('click', function(event) {
    if (event.target == bookModal || event.target == contentModal) {
        event.target.style.display = "none";
    }
});

// 책 표지 미리보기
function previewCover(files) {
    const preview = document.getElementById('coverPreview');
    preview.innerHTML = '';
    if (files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = new Image();
            img.src = e.target.result;
            img.className = 'preview-image';  // CSS로 스타일 지정 가능
            preview.appendChild(img);
        };
        reader.readAsDataURL(files[0]);
    }
}

// 책 내용 이미지 미리보기
function previewContent(files) {
    const preview = document.getElementById('contentPreview');
    preview.innerHTML = '';
    Array.from(files).forEach(file => {
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = new Image();
            img.src = e.target.result;
            img.className = 'preview-image';  // CSS로 스타일 지정 가능
            preview.appendChild(img);
        };
        reader.readAsDataURL(file);
    });
}

// 다음 단계로 이동 버튼 처리
document.querySelector('.btn.next').addEventListener('click', function(event) {
    event.preventDefault();
    bookModal.style.display = 'none';
    contentModal.style.display = 'block';
});

// 책 및 이미지 데이터 제출 처리
document.querySelector('.upload-content').addEventListener('click', function() {
    const formData = new FormData();
    const bookCover = document.getElementById('bookCover').files[0];
    const contentImages = document.getElementById('contentImages').files;

    formData.append('bookName', document.getElementById('bookName').value);
    formData.append('bookCover', bookCover);

    Array.from(contentImages).forEach((file, index) => {
        formData.append(`contentImage${index}`, file);
    });

    fetch('/addBookWithContent', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('Upload successful', data);
        window.location.reload();  // 성공 후 페이지 리로드
    })
    .catch(error => {
        console.error('Error:', error);
    });

    contentModal.style.display = 'none';  // 업로드 후 모달 닫기
});
