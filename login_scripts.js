document.addEventListener('DOMContentLoaded', function() {
    var showSignUp = document.getElementById('showSignUp');
    var findPW = document.getElementById("findPW");
    var showLoginFromSignup = document.getElementById('showLoginFromSignup');
    var showLoginFromFind = document.getElementById('showLoginFromFind');
    var loginForm = document.getElementById('loginForm');
    var signupForm = document.getElementById('signupForm');
    var findForm = document.getElementById('findForm');

    showSignUp.addEventListener('click', function() {
        loginForm.style.display = 'none';
        signupForm.style.display = 'block';
    });

    showLoginFromSignup.addEventListener('click', function() {
        signupForm.style.display = 'none';
        loginForm.style.display = 'block';
    });

    showLoginFromFind.addEventListener('click', function() {
        findForm.style.display = 'none';
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

    findPW.addEventListener('click', function() {
        loginForm.style.display = 'none';
        findForm.style.display = 'block';
    });
});
