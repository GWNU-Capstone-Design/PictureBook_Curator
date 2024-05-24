document.addEventListener("DOMContentLoaded", function() {
    var settingsIcon = document.getElementById("settingsIcon");
    var settingsModal = document.getElementById("settingsModal");
    var brightnessIcon = document.getElementById("brightnessIcon");
    var brightnessModal = document.getElementById("brightnessModal");
    var brightnessCloseBtn = brightnessModal.querySelector('.close');
    var pinIcon = document.getElementById("pinIcon");
    var restoreButton = document.getElementById("restoreButton");
    var header = document.querySelector("header");
    var footer = document.querySelector("footer");

    // 밝기 조절 모달 토글
    brightnessIcon.addEventListener('click', function() {
        brightnessModal.style.display = 'block';
    });

    // 밝기 모달 닫기
    brightnessCloseBtn.onclick = function() {
        brightnessModal.style.display = "none";
    };

    // 설정 모달 토글
    settingsIcon.onclick = function() {
        settingsModal.style.display = "block";
        settingsModal.classList.remove("hidden");
    };

    // 설정 모달 닫기
    var settingsCloseBtn = settingsModal.querySelector('.close');
    settingsCloseBtn.onclick = function() {
        settingsModal.style.display = "none";
        settingsModal.classList.add("hidden");
    };

    // 모달 외부 클릭시 닫기
    window.onclick = function(event) {
        if (event.target == settingsModal) {
            settingsModal.style.display = "none";
            settingsModal.classList.add("hidden");
            event.stopPropagation(); // 이벤트 버블링 방지
        }
        if (event.target == brightnessModal) {
            brightnessModal.style.display = "none";
            event.stopPropagation(); // 이벤트 버블링 방지
        }
    };

    pinIcon.addEventListener("click", function() {
        var isHidden = header.style.display === "none";
        header.style.display = isHidden ? "flex" : "none";
        footer.style.display = isHidden ? "flex" : "none";
        restoreButton.style.display = isHidden ? "none" : "flex";
    });

    restoreButton.addEventListener("click", function() {
        header.style.display = "flex";
        footer.style.display = "flex";
        restoreButton.style.display = "none";
    });
});
