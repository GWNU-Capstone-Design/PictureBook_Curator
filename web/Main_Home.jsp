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

<%
    // 세션에서 user_id 가져오기
    Integer userId = (Integer) request.getSession().getAttribute("user_id");


    // 책 표지 이미지 경로를 저장할 리스트
    List<String> bookCoverPaths = new ArrayList<>();
    Connection con = null;
    PreparedStatement pstmt = null;

    try {
        // DB 연결
        con = DatabaseConnector.getConnection();
        if (con != null && userId != null) {
            // SQL 쿼리 작성
            String query = "SELECT b.book_name, i.cover_image " +
                    "FROM book b " +
                    "JOIN image i ON b.book_id = i.book_id " +
                    "WHERE b.user_id = ?";

            // PreparedStatement 생성
            pstmt = con.prepareStatement(query);

            // 매개변수 설정
            pstmt.setInt(1, userId);

            // 쿼리 실행
            ResultSet rs = pstmt.executeQuery();

            while (rs.next()) {
                String bookName = rs.getString("book_name");
                String coverImage = rs.getString("cover_image");
                String imagePath = "image/" + bookName + "/" + coverImage;
                bookCoverPaths.add(imagePath);
            }
        }
    } catch (SQLException e) {
        // 오류 처리
        e.printStackTrace(); // 또는 로그에 오류를 기록합니다.
    } finally {
        // PreparedStatement 및 Connection 닫기
        if (pstmt != null) {
            try {
                pstmt.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
        if (con != null) {
            try {
                con.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }
%>

<script>
    const bookContainer = document.querySelector('.main-content');
    // 예시 이미지 파일 경로
    // 책 이미지 파일 경로 배열
    const imagePaths = [
        <% for (String imagePath : bookCoverPaths) { %>
        '<%= imagePath %>',
        <% } %>
    ];

    // 이미지 파일을 book 클래스에 추가
    imagePaths.forEach(path => {
        const bookDiv = document.createElement('div');
        bookDiv.classList.add('book');

        const img = document.createElement('img');
        img.src = path;

        bookDiv.appendChild(img);
        bookContainer.appendChild(bookDiv);

        img.addEventListener('click', () => {
            window.location.href = 'Viewer.html';
        });

    });

    // 책 추가 부분
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

    const bookForm = document.getElementById('bookForm');
    bookForm.addEventListener('submit', function(event) {
        //event.preventDefault();
        // 폼을 직접 제출하는 코드 추가
        this.submit();
    });
</script>

</body>
</html>
