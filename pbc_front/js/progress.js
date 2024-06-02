let progress = document.getElementById("progress");
let song = document.getElementById("song");
let ctrlIcon = document.getElementById("ctrlIcon");
let vol = document.getElementById("vol");
const volumeIcon = document.querySelector('.volume-icon i');
let isMuted = false;
let previousVolume = song.volume; // Store the previous volume before mute

// Update the progress bar when the song's metadata is loaded
song.onloadedmetadata = function() {
    progress.max = song.duration;
    progress.value = song.currentTime;
}

// Play or pause the song when the play/pause button is clicked
function playPause() {
    if (ctrlIcon.classList.contains("bx-stop")) {
        song.pause();
        ctrlIcon.classList.remove("bx-stop");
        ctrlIcon.classList.add("bx-play");
    } else {
        song.play();
        ctrlIcon.classList.add("bx-stop");
        ctrlIcon.classList.remove("bx-play");
    }
}

// Update the progress bar as the song plays
if (song.play()) {
    setInterval(() => {
        progress.value = song.currentTime;
    }, 500);
}

// Seek the song to the position of the progress bar when changed
progress.onchange = function() {
    song.currentTime = progress.value;
    // 노래가 재생 중이면 멈추지 않고 계속 재생
    if (!song.paused) {
        song.play();
        ctrlIcon.classList.add("bx-stop");
        ctrlIcon.classList.remove("bx-play");
    }
}

// Rewind the song by 5 seconds
function rewind() {
    song.currentTime = Math.max(0, song.currentTime - 5);
}

// Fast forward the song by 5 seconds
function fastForward() {
    song.currentTime = Math.min(song.duration, song.currentTime + 5);
}

// Add event listeners to the rewind and fast forward buttons
document.querySelector('.bx-rewind').addEventListener('click', rewind);
document.querySelector('.bx-fast-forward').addEventListener('click', fastForward);

vol.oninput = function() {
    if (isMuted) {
        volumeIcon.classList.remove('bx-volume-mute');
        volumeIcon.classList.add('bx-volume-full');
        isMuted = false;
    }
    song.volume = vol.value / 100;
    previousVolume = song.volume; // Update previous volume
}

volumeIcon.addEventListener('click', () => {
    if (volumeIcon.classList.contains('bx-volume-full')) {
        previousVolume = song.volume; // Store the current volume
        song.volume = 0; // Mute the volume
        volumeIcon.classList.remove('bx-volume-full');
        volumeIcon.classList.add('bx-volume-mute');
        isMuted = true;
    } else {
        song.volume = previousVolume; // Restore the previous volume
        volumeIcon.classList.remove('bx-volume-mute');
        volumeIcon.classList.add('bx-volume-full');
        isMuted = false;
    }
});

// Bookmark
const bookmarkButton = document.querySelector('.bookmark-button i');

bookmarkButton.addEventListener('click', () => {
    if (bookmarkButton.classList.contains('bx-bookmark-plus')) {
        bookmarkButton.classList.remove('bx-bookmark-plus');
        bookmarkButton.classList.add('bx-bookmark-minus');
    } else {
        bookmarkButton.classList.remove('bx-bookmark-minus');
        bookmarkButton.classList.add('bx-bookmark-plus');
    }
});