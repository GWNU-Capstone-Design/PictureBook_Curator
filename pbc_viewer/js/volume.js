// Volume/Mute toggle logic
const volumeIcon = document.querySelector('.volume-icon i');

volumeIcon.addEventListener('click', () => {
    if (volumeIcon.classList.contains('bx-volume-full')) {
        volumeIcon.classList.remove('bx-volume-full');
        volumeIcon.classList.add('bx-volume-mute');
    } else {
        volumeIcon.classList.remove('bx-volume-mute');
        volumeIcon.classList.add('bx-volume-full');
    }
});