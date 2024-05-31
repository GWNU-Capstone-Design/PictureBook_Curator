document.addEventListener('DOMContentLoaded', () => {
    const addBookButton = document.querySelector('.add-book');
    const modal = document.getElementById('addBookModal');
    const viewerModal = document.getElementById('viewerModal');
    const closeBtns = document.querySelectorAll('.close');
    const nextStepButton = document.getElementById('nextStep');
    const addBookFinalButton = document.getElementById('addBook');
    const coverImageInput = document.getElementById('coverImage');
    const coverPreview = document.getElementById('coverPreview');
    const contentImagesInput = document.getElementById('contentImages');
    const contentPreviewContainer = document.getElementById('contentPreviewContainer');
    const step1 = document.getElementById('step1');
    const step2 = document.getElementById('step2');
  
    const viewerTitle = document.getElementById('viewerTitle');
    const viewerImage = document.getElementById('viewerImage');
    const currentPageElem = document.getElementById('currentPage');
    const totalPagesElem = document.getElementById('totalPages');
  
    let books = [];
    let currentBook = null;
    let currentPage = 1;
  
    // 함수 정의
    function resetModal() {
      step1.style.display = 'block';
      step2.style.display = 'none';
      document.getElementById('bookTitle').value = '';
      coverImageInput.value = '';
      coverPreview.src = '#';
      coverPreview.style.display = 'none';
      contentImagesInput.value = '';
      contentPreviewContainer.innerHTML = '';
    }
  
    function addBookToLibrary(book) {
      const bookContainer = document.createElement('div');
      bookContainer.classList.add('book');
      const img = document.createElement('img');
      img.src = book.cover;
      img.alt = book.title;
      bookContainer.appendChild(img);
      bookContainer.addEventListener('click', () => {
        openBook(book);
      });
      document.querySelector('.main-content').appendChild(bookContainer);
    }
  
    function openBook(book) {
      currentBook = book;
      currentPage = 1;
      viewerTitle.textContent = book.title;
      viewerImage.src = book.pages[0];
      currentPageElem.textContent = currentPage;
      totalPagesElem.textContent = book.pages.length;
      viewerModal.style.display = 'block';
    }
  
    // 이벤트 리스너 설정
    addBookButton.addEventListener('click', () => {
      modal.style.display = 'block';
    });
  
    closeBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        modal.style.display = 'none';
        viewerModal.style.display = 'none';
        resetModal();
      });
    });
  
    window.addEventListener('click', (event) => {
      if (event.target === modal) {
        modal.style.display = 'none';
        resetModal();
      }
      if (event.target === viewerModal) {
        viewerModal.style.display = 'none';
      }
    });
  
    nextStepButton.addEventListener('click', () => {
      step1.style.display = 'none';
      step2.style.display = 'block';
    });
  
    coverImageInput.addEventListener('change', (event) => {
      const file = event.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          coverPreview.src = e.target.result;
          coverPreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
      }
    });
  
    contentImagesInput.addEventListener('change', (event) => {
      contentPreviewContainer.innerHTML = '';
      const files = event.target.files;
      if (files) {
        Array.from(files).forEach(file => {
          const reader = new FileReader();
          reader.onload = (e) => {
            const img = document.createElement('img');
            img.src = e.target.result;
            contentPreviewContainer.appendChild(img);
          };
          reader.readAsDataURL(file);
        });
      }
    });
  
    addBookFinalButton.addEventListener('click', () => {
      const bookTitle = document.getElementById('bookTitle').value;
      const coverImage = coverImageInput.files[0];
      const contentImages = contentImagesInput.files;
  
      if (!bookTitle || !coverImage || contentImages.length === 0) {
        alert('모든 필드를 입력해주세요.');
        return;
      }
  
      const formData = new FormData();
      formData.append('bookTitle', bookTitle);
      formData.append('coverImage', coverImage);
      Array.from(contentImages).forEach((file, index) => {
        formData.append(`contentImages`, file);
      });
  
      fetch('/upload', {
        method: 'POST',
        body: formData
      })
      .then(response => response.json())
      .then(data => {
        if (data.success) {
          books.push(data.book);
          addBookToLibrary(data.book);
        } else {
          alert('책 추가에 실패했습니다.');
        }
      })
      .catch(error => {
        console.error('Error:', error);
        alert('오류가 발생했습니다.');
      });
  
      modal.style.display = 'none';
      resetModal();
    });
  
    document.getElementById('prevPage').addEventListener('click', () => {
      if (currentPage > 1) {
        currentPage--;
        viewerImage.src = currentBook.pages[currentPage - 1];
        currentPageElem.textContent = currentPage;
      }
    });
  
    document.getElementById('nextPage').addEventListener('click', () => {
      if (currentPage < currentBook.pages.length) {
        currentPage++;
        viewerImage.src = currentBook.pages[currentPage - 1];
        currentPageElem.textContent = currentPage;
      }
    });
  });