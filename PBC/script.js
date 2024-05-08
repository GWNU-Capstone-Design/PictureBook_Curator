document.addEventListener("DOMContentLoaded", function() {
    const pinIcon = document.getElementById("pinIcon");
    const restoreButton = document.getElementById("restoreButton");
    const header = document.querySelector("header");
    const footer = document.querySelector("footer");

    pinIcon.addEventListener("click", function() {
        header.style.display = "none";
        footer.style.display = "none";
        restoreButton.style.display = "block"; // 복원 버튼을 보이게 함
    });

    restoreButton.addEventListener("click", function() {
        header.style.display = "flex";
        footer.style.display = "flex";
        restoreButton.style.display = "none"; // 복원 버튼을 숨김
    });
});
