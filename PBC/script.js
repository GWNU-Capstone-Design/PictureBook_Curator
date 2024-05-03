document.addEventListener("DOMContentLoaded", function() {
    const pinIcon = document.getElementById("pinIcon");
    const header = document.querySelector("header");
    const footer = document.querySelector("footer");

    pinIcon.addEventListener("click", function() {
        // 헤더와 푸터 숨기기
        header.style.display = "none";
        footer.style.display = "none";
    });
});
