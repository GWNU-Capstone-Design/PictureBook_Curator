document.addEventListener('DOMContentLoaded', function() {
    // 슬라이드 요소를 선택합니다.
    const slides = document.querySelectorAll('input[type="radio"][name="slide"]');
    // 자막을 업데이트할 요소를 선택합니다.
    const subtitle = document.querySelector('.subtitle');
    // 현재 페이지 입력 요소를 선택합니다.
    const currentPageInput = document.getElementById('current-page');
    // 전체 페이지 수를 슬라이드 개수로 설정합니다.
    const totalPages = slides.length;

    // 총 페이지 수를 두 자리 형식으로 표시합니다.
    document.getElementById('total-pages').textContent = totalPages.toString().padStart(2, '0');

    // 슬라이드 변경 이벤트에 대한 리스너를 추가합니다.
    slides.forEach(slide => {
        slide.addEventListener('change', function() {
            updateSubtitle();
            updateCurrentPage();
        });
    });

    // 현재 페이지 입력 값 변경 이벤트에 대한 리스너를 추가합니다.
    currentPageInput.addEventListener('change', function() {
        const page = parseInt(currentPageInput.value);
        if (page >= 1 && page <= totalPages) {
            document.getElementById(`slide0${page}`).checked = true;
            updateSubtitle();
        } else {
            currentPageInput.value = formatPageNumber(document.querySelector('input[type="radio"][name="slide"]:checked').id.replace('slide', ''));
        }
    });

    // 자막을 업데이트하는 함수입니다.
    function updateSubtitle() {
        const checkedSlide = document.querySelector('input[type="radio"][name="slide"]:checked').id;
        const currentPage = checkedSlide.replace('slide', '');
        currentPageInput.value = formatPageNumber(currentPage); // 현재 페이지 번호 업데이트
        const subtitles = {
            slide01: "첫번째 슬라이드 입니다. 이 문장은 자막이 길어질 때 어떻게 보이는지 테스트하는 예시 문장입니다.",
            slide02: "두번째 슬라이드 입니다. 두 번째 슬라이드는 길이 테스트를 위해 작성되었습니다. 이 문장도 충분히 길어야 합니다.",
            slide03: "세번째 슬라이드 입니다. 이 슬라이드는 자막이 길어질 때의 처리를 확인하기 위해 작성되었습니다. 충분히 기이이이이이인 문장입니다."
        };
        const subtitleText = subtitles[checkedSlide];
        subtitle.textContent = subtitleText;

        adjustFontSize(subtitle, subtitleText);
    }

    // 현재 페이지 번호를 업데이트하는 함수입니다.
    function updateCurrentPage() {
        const checkedSlide = document.querySelector('input[type="radio"][name="slide"]:checked').id;
        const currentPage = checkedSlide.replace('slide', '');
        currentPageInput.value = formatPageNumber(currentPage);
    }

    // 페이지 번호를 두 자리 형식으로 변환하는 함수입니다.
    function formatPageNumber(page) {
        return page.padStart(2, '0');
    }

    // 자막의 글자 크기를 조정하는 함수입니다.
    function adjustFontSize(element, text) {
        const maxFontSize = 16; // 최대 글자 크기
        const minFontSize = 12; // 최소 글자 크기
        const maxLength = 50; // 기준 텍스트 길이

        // 텍스트 길이에 따라 글자 크기 조정
        const textLength = text.length;
        const fontSize = textLength > maxLength 
            ? Math.max(minFontSize, maxFontSize - (textLength - maxLength) * 0.1) 
            : maxFontSize;
        element.style.fontSize = fontSize + 'px';
    }

    // 자막이 컨텐츠 영역을 벗어나는지 확인하는 함수입니다.
    function checkSubtitleOverflow() {
        const content = document.querySelector('.content');
        if (subtitle.offsetHeight + subtitle.offsetTop > content.offsetHeight) {
            subtitle.style.visibility = 'hidden'; // 자막이 컨텐츠 영역을 벗어나면 숨김
        } else {
            subtitle.style.visibility = 'visible'; // 자막이 컨텐츠 영역에 맞으면 보임
        }
    }

    // 초기 자막 및 페이지 업데이트
    updateSubtitle();  // 초기 상태에서 자막을 설정합니다.
    updateCurrentPage();  // 초기 상태에서 페이지 번호를 설정합니다.
});

// Play/Stop toggle logic
const playButton = document.querySelector('.pause-play-toggle i');

playButton.addEventListener('click', () => {
    if (playButton.classList.contains('bx-play-circle')) {
        playButton.classList.remove('bx-play-circle');
        playButton.classList.add('bx-stop-circle');
    } else {
        playButton.classList.remove('bx-stop-circle');
        playButton.classList.add('bx-play-circle');
    }
});