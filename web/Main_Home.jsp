<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%@ page import="java.sql.*, java.util.ArrayList, java.util.List" %>
<%@ page import="Database.DatabaseConnector" %>
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="css/nnew_styles.css">
    <link href='https://unpkg.com/boxicons@2.1.4/css/boxicons.min.css' rel='stylesheet'>
    <link rel="icon" type="image/png" href="image/pbc_logo1.png">
    <title>PictureBook Curator</title>
</head>
<body>

<div class="header">
    <div class="title">
        <a href="new_index.html">
            <img src="image/pbc_logo2.png" width="200px">
        </a>
    </div>
    <div class="icons">
        <span class="icon search-icon"></span>
        <span class="icon notification-icon"></span>
        <span class="icon settings-icon"></span>
    </div>
</div>

<div class="sorting-tabs">
    <div class="tab">날짜</div>
    <div class="tab">이름</div>
    <div class="tab">유형</div>
</div>

<div class="main-content">
    <div class="add-book">+</div>
</div>

<div id="addBookModal" class="modal">
    <div class="modal-content">
        <span class="close">&times;</span>
        <form action="UploadCoverServlet" method="post" enctype="multipart/form-data" id="bookForm" accept-charset="UTF-8">
            <h1>책 정보 업로드</h1>
            <div class="input-box">
                <label for="bookName">제목</label>
                <input type="text" id="bookName" name="bookName" required>
                <i class='bx bx-book'></i>
            </div>

            <label for="bookCover">표지</label>
            <input type="file" class="upload-box" id="bookCover" name="bookCover" accept="image/*" onchange="previewCover(this.files)" required>
            <div id="coverPreview"></div><br>

            <button type="submit" class="btn"><i class='bx bxs-plus-square'></i>&nbsp;&nbsp;책 추가</button>
        </form>
    </div>
</div>
<script>
    // 이미지 경로를 저장할 Set
    let imagePaths = new Set();

    document.addEventListener('DOMContentLoaded', function() {
        // 페이지 로드 후 처음 한 번 데이터를 불러옵니다.
        fetchBookCoverPaths();

        // 일정 시간 간격으로 데이터를 업데이트합니다.
        setInterval(fetchBookCoverPaths, 5000); // 5초마다 업데이트
    });

    function fetchBookCoverPaths() {
        // AJAX를 사용하여 서버로부터 데이터를 요청합니다.
        var xhr = new XMLHttpRequest();
        xhr.open('GET', 'MainPageData.jsp', true);
        xhr.onreadystatechange = function() {
            if (xhr.readyState === XMLHttpRequest.DONE) {
                if (xhr.status === 200) {
                    // 요청이 성공적으로 완료되었을 때 데이터를 처리합니다.
                    var data = JSON.parse(xhr.responseText);
                    updatePageWithData(data);
                } else {
                    // 요청이 실패했을 때 오류를 처리합니다.
                    console.error('서버에서 오류 발생:', xhr.status);
                }
            }
        };
        xhr.send();
    }

    function updatePageWithData(data) {
        // 받아온 데이터를 사용하여 페이지를 업데이트합니다.
        displayBookCovers(data);
    }

    // 책 표지 이미지를 표시하는 함수
    function displayBookCovers(paths) {
        const bookContainer = document.querySelector('.main-content');

        paths.forEach(path => {
            // 이미지 경로가 이미 존재하는지 확인
            if (!imagePaths.has(path)) {
                const bookDiv = document.createElement('div');
                bookDiv.classList.add('book');

                const img = document.createElement('img');
                img.src = path;

                bookDiv.appendChild(img);
                bookContainer.appendChild(bookDiv);

                img.addEventListener('click', () => {
                    window.location.href = 'Viewer.jsp';
                });

                // 이미지 경로를 Set에 추가
                imagePaths.add(path);
            }
        });
    }

    // 책 추가 모달 관련 코드
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

    const bookForm = document.getElementById('bookForm');
    bookForm.addEventListener('submit', function(event) {
        //event.preventDefault();
        // 폼을 직접 제출하는 코드 추가
        this.submit();
    });
</script>
</body>
</html>
