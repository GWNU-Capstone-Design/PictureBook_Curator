document.addEventListener('DOMContentLoaded', function() {
    var showSignUp = document.getElementById('showSignUp');
    var showLogin = document.getElementById('showLogin');
    var loginForm = document.getElementById('loginForm');
    var signupForm = document.getElementById('signupForm');

    showSignUp.addEventListener('click', function() {
        loginForm.style.display = 'none';
        signupForm.style.display = 'block';
    });

    showLogin.addEventListener('click', function() {
        signupForm.style.display = 'none';
        loginForm.style.display = 'block';
    });

    loginForm.addEventListener('submit', function(event) {
        event.preventDefault();
        // 로그인 로직
    });

    signupForm.addEventListener('submit', function(event) {
        event.preventDefault();
        // 회원가입 로직
    });
});