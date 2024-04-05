<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>로그인</title>
  <link rel="stylesheet" href="css/login_styles.css">
</head>
<body>

<div class="form-container">
  <form action="login_Check.jsp" id="loginForm" method="post">
    <h2>로그인</h2>
    <input type="email" id="loginEmail" placeholder="이메일" required>
    <input type="password" id="loginPassword" placeholder="비밀번호" required>
    <button type="submit" onclick="location.href = 'login_Check.jsp'">로그인</button>
    <p><span id="findPW">비밀번호찾기</span> <span id="showSignUp" onclick="location.href = 'SignUp.jsp';">회원가입</span></p>
  </form>

  <form id="findForm" style="display:none;">
    <h2>비밀번호찾기</h2>
    <input type="email" id="signup_Email" placeholder="이메일" required>
    <button type="submit">비밀번호 찾기</button>
    <p>다시 로그인하기  <span id="show_Login">로그인</span></p>
  </form>
</div>
</body>
</html>
