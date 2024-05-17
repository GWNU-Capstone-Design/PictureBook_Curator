document.addEventListener("DOMContentLoaded", function() {
    var settingsIcon = document.getElementById("settingsIcon");
    var settingsModal = document.getElementById("settingsModal");
    var brightnessIcon = document.getElementById("brightnessIcon");
    var brightnessContainer = document.querySelector(".brightness-control-container");
    var pinIcon = document.getElementById("pinIcon");
    var restoreButton = document.getElementById("restoreButton");
    var header = document.querySelector("header");
    var footer = document.querySelector("footer");

    // Settings Modal Controls
    settingsIcon.onclick = function() {
        settingsModal.style.display = "block";
        settingsModal.classList.remove("hidden");  // 'hidden' 클래스 제거
    };

    // Correct close button for the settings modal
    var settingsCloseBtn = settingsModal.querySelector('.close');
    settingsCloseBtn.onclick = function() {
        settingsModal.style.display = "none";
        settingsModal.classList.add("hidden");  // 'hidden' 클래스 추가
    };

    window.onclick = function(event) {
        if (event.target == settingsModal) {
            settingsModal.style.display = "none";
            settingsModal.classList.add("hidden");  // 'hidden' 클래스 추가
        }
    };

    // Brightness Control Visibility
    brightnessIcon.addEventListener("click", function() {
        brightnessContainer.style.display = (brightnessContainer.style.display == "flex") ? "none" : "flex";
    });

    // Pin (Hide/Show UI Elements)
    pinIcon.addEventListener("click", function() {
        header.style.display = "none";
        footer.style.display = "none";
        restoreButton.style.display = "block"; // Display the restore button
    });

    // Restore UI Elements
    restoreButton.addEventListener("click", function() {
        header.style.display = "flex";
        footer.style.display = "flex";
        restoreButton.style.display = "none"; // Hide the restore button again
    });

    // Back button action
    function goBack() {
        window.history.back();
    }
});
