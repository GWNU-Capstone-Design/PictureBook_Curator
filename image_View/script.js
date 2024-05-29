/* 브라우저 기본 마진과 패딩 리셋 */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box; /* 모든 요소에 박스 모델 적용 */
}

body {
    height: 100vh; /* 화면 전체 높이 */
    display: flex; /* 플렉스 박스 레이아웃 사용 */
    justify-content: center; /* 수평 중앙 정렬 */
    align-items: center; /* 수직 중앙 정렬 */
    font-family: Arial, sans-serif; /* 폰트 설정 */
    background-color: #f0f0f0; /* 배경색 설정 */
}

.container {
    border: 2px solid #000; /* 검정색 테두리 */
    width: 80%; /* 너비 80% */
    height: 80%; /* 높이 80% */
    display: flex;
    flex-direction: column; /* 세로 방향 정렬 */
    justify-content: space-between; /* 공간을 균등하게 분배 */
    align-items: center; /* 수평 중앙 정렬 */
    background-color: #fff; /* 배경색 흰색 */
}

.header, .footer {
    display: flex;
    justify-content: space-between; /* 양 끝에 배치 */
    align-items: center; /* 수직 중앙 정렬 */
    width: 100%; /* 전체 너비 */
    padding: 10px; /* 패딩 설정 */
}

.header {
    border-bottom: 2px solid #000; /* 하단 테두리 */
}

.footer {
    border-top: 2px solid #000; /* 상단 테두리 */
}

.content {
    display: flex;
    justify-content: center; /* 수평 중앙 정렬 */
    align-items: center; /* 수직 중앙 정렬 */
    flex-direction: column; /* 세로 방향 정렬 */
    width: 100%; /* 전체 너비 */
    height: 80%; /* 전체 높이 */
    position: relative; /* 상대 위치 */
}

/* 슬라이드 스타일 */
.section input[id*="slide"] {
    display: none;
}
/* 이미지 크기 변경 */
.slidewrap {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
}

.section .slidewrap {
    max-width: 100%;
    margin: 0 auto;
    overflow: hidden;
}

.section .slidelist {
    white-space: nowrap;
    font-size: 0;
}

.section .slidelist > li {
    display: inline-block;
    vertical-align: middle;
    width: 100%;
    height: 100%;
    transition: all 0.5s;
}

.section .slidelist > li > a {
    display: block;
    position: relative;
    overflow: hidden; /* 화살표 화면 밖으로 나가면 안 보이도록 overflow로 가림처리 */
}

.section .slidelist > li > a img {
    width: 100%;
    height: 100%;
    object-fit: cover; /* 이미지가 컨테이너의 너비와 높이에 맞춰 축소 */

}

/* 좌우로 넘기는 LABEL버튼에 대한 스타일 */
.section .slidelist label {
    position: absolute;
    z-index: 1;
    top: 50%;
    transform: translateY(-50%);
    padding: 0px;
    cursor: pointer;
    width: 50px; /* 화살표 크기를 적당히 설정 */
    height: 50px; /* 동일한 크기로 설정하여 원형을 유지 */
    background: rgba(0, 0, 0, 0); /* 초기 배경은 완전 투명 */
    transition: background-color 0.3s; /* 부드러운 배경색 변경 효과 */
    border-radius: 50%; /* 원형 모양으로 만듦 */
    display: flex;
    justify-content: center;
    align-items: center;
}

/* 마우스를 올렸을 때의 스타일 */
.section .slidelist label:hover {
    background-color: rgba(0, 0, 0, 0.3); /* 마우스 오버시 반투명 검정색 배경 추가 */
    opacity: 1; /* 화살표를 보이게 함 */
}

.section .slidelist .left {
    left: 50px; /* 화면 왼쪽 가장자리에서 적당히 떨어진 위치 */
    background: url('./img/left.png') center center / contain no-repeat;
}

.section .slidelist .right {
    right: 50px; /* 화면 오른쪽 가장자리에서 적당히 떨어진 위치 */
    background: url('./img/right.png') center center / contain no-repeat;
}

/* INPUT이 체크되면 변화값이 li까지 전달되는 스타일 */
.section input[id="slide01"]:checked ~ .slidewrap .slidelist > li {
    transform: translateX(0%);
}

.section input[id="slide02"]:checked ~ .slidewrap .slidelist > li {
    transform: translateX(-100%);
}

.section input[id="slide03"]:checked ~ .slidewrap .slidelist > li {
    transform: translateX(-200%);
}

/* INPUT이 체크되면 변화값이 LEFT,RIGHT에 전달되는 스타일 */
.section input[id="slide01"]:checked ~ .slidewrap li:nth-child(1) .left {
    left: 0px;
    transition: all 0.35s ease 0.5s;
}

.section input[id="slide01"]:checked ~ .slidewrap li:nth-child(1) .right {
    right: 0px;
    transition: all 0.35s ease 0.5s;
}

.section input[id="slide02"]:checked ~ .slidewrap li:nth-child(2) .left {
    left: 0px;
    transition: all 0.35s ease 0.5s;
}

.section input[id="slide02"]:checked ~ .slidewrap li:nth-child(2) .right {
    right: 0px;
    transition: all 0.35s ease 0.5s;
}

.section input[id="slide03"]:checked ~ .slidewrap li:nth-child(3) .left {
    left: 0px;
    transition: all 0.35s ease 0.5s;
}

.section input[id="slide03"]:checked ~ .slidewrap li:nth-child(3) .right {
    right: 0px;
    transition: all 0.35s ease 0.5s;
}

/* 버튼 크기 및 제어 */
button {
    border: none; /* 테두리 제거 */
    background-color: transparent; /* 배경색 투명 */
    cursor: pointer; /* 커서 모양 변경 */
    margin: 0; /* 여백 설정 */
    font-size: 25px;
    transition: transform 0.5s; /* 변환 시 애니메이션 적용 */
}


.back-button, .bookmark-button, .action-button {
    background: none; /* 배경 없음 */
    border: none; /* 테두리 없음 */
    font-size: 18px; /* 폰트 크기 */
    cursor: pointer; /* 커서 모양 변경 */
    display: flex; /* 플렉스 박스 사용 */
    align-items: center; /* 수직 중앙 정렬 */
}

.back-button img, .bookmark-button img, .action-button img {
    width: 24px; /* 이미지 너비 */
    height: 24px; /* 이미지 높이 */
}

.title-input {
    border: 1px solid #ccc; /* 테두리 색상 */
    border-radius: 4px; /* 테두리 둥글게 */
    padding: 5px; /* 패딩 설정 */
    font-size: 18px; /* 폰트 크기 */
    text-align: center; /* 텍스트 중앙 정렬 */
    width: 60%; /* 너비 설정 */
}

.controls {
    display: flex; /* 플렉스 박스 사용 */
    justify-content: center; /* 중앙 정렬 */
    align-items: center; /* 수직 중앙 정렬 */
    flex-grow: 1; /* 남는 공간 차지 */
}

.control-button {
    margin: 0 10px; /* 좌우 여백 설정 */
    background: none; /* 배경 없음 */
    border: none; /* 테두리 없음 */
    font-size: 18px; /* 폰트 크기 */
    cursor: pointer; /* 커서 모양 변경 */
}

.control-button img.icon {
    width: 24px; /* 아이콘 너비 */
    height: 24px; /* 아이콘 높이 */
    vertical-align: middle; /* 수직 중앙 정렬 */
}

.volume-control {
    display: flex; /* 플렉스 박스 사용 */
    align-items: center; /* 수직 중앙 정렬 */
}

.volume-icon {
    margin-right: 10px; /* 오른쪽 여백 */
}

.volume-icon img {
    width: 24px; /* 아이콘 너비 */
    height: 24px; /* 아이콘 높이 */
    cursor: pointer; /* 커서 모양 변경 */
}

.volume-control input {
    cursor: pointer; /* 커서 모양 변경 */
}

/* 자막 스타일 추가 */
.subtitle {
    border: 2px solid black;
    position: absolute;
    bottom: 10px; /* 하단에서 약간 위로 띄움 */
    padding: 5px 10px;
    background: rgba(255, 255, 255, 1); /* 반투명 배경 */
    color: black; /* 글자 색상 */
    font-size: 1rem;
    text-align: center;
    border-radius: 5px;
    font-weight: bold; /* 글자 두껍게 */
    white-space: normal; /* 줄 바꿈을 허용 */
    word-wrap: break-word; /* 긴 단어를 줄 바꿈 */
    max-width: 90%; /* 최대 너비 설정 */
    min-width: 100px; /* 최소 너비 설정 */
    transition: all 0.3s ease-in-out; /* 부드러운 애니메이션 */
    display: inline-block; /* 텍스트 길이에 맞춰 배경 크기 조정 */
}

/* 자막의 글자 크기를 조절하는 스타일 추가 */
.subtitle.adjust-font {
    font-size: calc(1rem + 0.5vw);
}

.page-info {
    display: flex;
    align-items: center;
    font-size: 16px;
    color: black;
    margin: 0 30px; /* 좌우 여백을 추가하여 가운데 배치 */
}

.page-info input[type="number"] {
    width: 60px;
    margin-right: 5px;
    text-align: center;
    border: 1px solid #ccc;
    border-radius: 4px;
    padding: 5px;
    font-size: 16px;
}

