document.addEventListener("DOMContentLoaded", function() {
    const titleInput = document.getElementById("title-input");

    titleInput.addEventListener("input", function() {
        document.title = titleInput.value || "책 제목";
    });
});

// References to Dom Elements
const prevBtn = document.querySelector("#prev-btn");
const nextBtn = document.querySelector("#next-btn");
const book = document.querySelector("#book");

const paper1 = document.querySelector("#p1");
const paper2 = document.querySelector("#p2");
const paper3 = document.querySelector("#p3");
const paper4 = document.querySelector("#p4"); // 추가된 부분

const subtitle = document.getElementById("subtitle"); // 자막 요소 참조

// Event Listener
prevBtn.addEventListener("click", goPrevPage);
nextBtn.addEventListener("click", goNextPage);

// Business Logic
let currentLocation = 1;
let numOfPapers = 4; // 업데이트된 부분
let maxLocation = numOfPapers + 1;

function openBook() {
    book.style.transform = "translateX(50%)";
    prevBtn.style.transform = "translateX(-180px)";
    nextBtn.style.transform = "translateX(180px)";
}

function closeBook(isAtBeginning) {
    if (isAtBeginning) {
        book.style.transform = "translateX(0%)";
    } else {
        book.style.transform = "translateX(100%)";
    }
    prevBtn.style.transform = "translateX(0px)";
    nextBtn.style.transform = "translateX(0px)";
}

function updateSubtitle(text) {
    subtitle.innerText = text;

    // 자막 글자 크기를 조절
    subtitle.classList.remove("adjust-font");
    if (subtitle.scrollWidth > subtitle.clientWidth) {
        subtitle.classList.add("adjust-font");
    }
}

function goNextPage() {
    if (currentLocation < maxLocation) {
        switch (currentLocation) {
            case 1:
                openBook();
                paper1.classList.add("flipped");
                paper1.style.zIndex = 1;
                updateSubtitle("Front 1");
                break;
            case 2:
                paper2.classList.add("flipped");
                paper2.style.zIndex = 2;
                updateSubtitle("Front 2");
                break;
            case 3:
                paper3.classList.add("flipped");
                paper3.style.zIndex = 3;
                updateSubtitle("Front 3");
                break;
            case 4:
                paper4.classList.add("flipped");
                paper4.style.zIndex = 4;
                closeBook(false);
                updateSubtitle("Front 4");
                break;
            default:
                throw new Error("unknown state");
        }
        currentLocation++;
    }
}

function goPrevPage() {
    if (currentLocation > 1) {
        switch (currentLocation) {
            case 2:
                closeBook(true);
                paper1.classList.remove("flipped");
                paper1.style.zIndex = 4;
                updateSubtitle("Front 1");
                break;
            case 3:
                paper2.classList.remove("flipped");
                paper2.style.zIndex = 3;
                updateSubtitle("Front 2");
                break;
            case 4:
                paper3.classList.remove("flipped");
                paper3.style.zIndex = 2;
                updateSubtitle("Front 3");
                break;
            case 5:
                openBook();
                paper4.classList.remove("flipped");
                paper4.style.zIndex = 1;
                updateSubtitle("Front 4");
                break;
            default:
                throw new Error("unknown state");
        }
        currentLocation--;
    }
}

// Pause/Play toggle logic
const controlButton = document.querySelector('.pause-play-toggle img');
    
controlButton.addEventListener('click', () => {
    if (controlButton.getAttribute('src') === 'Icon/pause.png') {
        controlButton.setAttribute('src', 'Icon/play.png');
        controlButton.setAttribute('alt', 'Play');
    } else {
        controlButton.setAttribute('src', 'Icon/pause.png');
        controlButton.setAttribute('alt', 'Pause');
    }
});

// Volume/Mute toggle logic
const volumeIcon = document.getElementById('volume-icon');

volumeIcon.addEventListener('click', () => {
    if (volumeIcon.getAttribute('src') === 'Icon/volume.png') {
        volumeIcon.setAttribute('src', 'Icon/mute.png');
        volumeIcon.setAttribute('alt', 'Mute');
    } else {
        volumeIcon.setAttribute('src', 'Icon/volume.png');
        volumeIcon.setAttribute('alt', 'Volume');
    }
});
