# PictureBook_Curator
강릉원주대학교 컴퓨터공학과 2024년 1학기 캡스톤디자인I 4조

## 📖 프로젝트 소개
그림책 이미지를 사진 찍으면 이미지캡셔닝을 통해 그림을 설명하는 키워드를 추출합니다.

GPTs를 활용하여 이미지 내용을 설명하는 시나리오를 생성합니다.

생성한 시나리오를 TTS를 통해 목소리로 들려주는 프로젝트입니다.

## 🔗 링크
http://39.125.174.224:8080/PictureBook_Curator_war_exploded/

(서버 열릴 시 접근 가능)

## 🕰️개발 기간
- 2024.03.04.(월) ~ 2024.04.27.(금)
- 2024.03.15. 1차 회의: 주제 구체화 선정 및 구체화 회의
- 2024.03.17. 2차 회의: 프로젝트 제안서 작성 및 다이어그램 제작
- 2024.03.29. 3차 회의: 캡스톤 디자인 신청서 작성 및 개발 방향
- 2024.04.05. 4차 회의: PictureBook Curator 개발 1차
	- 이미지캡셔닝: 모델을 이용하여 사진 분류 및 CSV 파일 키워드 제작, 전처리 오류 해결하기
	- DB: 테이블 제작 및 프론트 기능 구현
	- Web: 인터페이스 및 기능 구현
- 2024.04.12. 5차 회의: PictureBook Curator 개발 2차
	- 이미지캡셔닝: 개발 도중 발생한 문제를 해결하기 위한 대안 모색
	- DB: 효율적인 테이블 제작, DB 어떤 모델을 사용할지 의논
	- Web: 웹에서 카메라 기능 구현, TTS 기능 개발
- 2024.04.19. 6차 회의: PictureBook Curator 개발 확인 및 피드백
	- 이미지캡셔닝: 한글 이미지캡셔닝 훈련 후 모델 업로드, 한글 자료 수 늘려서 키워드 받아오기
	- DB: 테이블 구축 및 로그인 시스템 개발, 회원가입/비밀번호 변경 구현
	- Web: 로그인 인터페이스 및 스토리보드 구현, 추가적인 기능 개발 예정, TTS 개발 완료
- 2024.04.26. 7차 회의: 중간 점검

## 🧑‍💻프로그래머 소개
- **김태겸** : 팀장, AI 학습 및 개발
- **김동찬** : AI 학습 및 개발
- **신용선** : DB 관리 및 설계
- **신지혜** : 웹사이트 및 앱 개발
- **전재영** : 웹사이트 및 앱 개발, TTS 기능 개발
- **정효진** : DB 관리 및 설계

## 💻 개발 환경
- **이미지 캡셔닝** : Google Colab 혹은 VScode
- **DB** : IntelliJ, MySQL
- **웹** : VScode
- **TTS** : Google TTS

## 🗨️ 파일 설명
### Branch : SYS (Branch : JJY)
- **web/css/login_style.css** : Find_pw.html, Reset_pw.html, SignUp.html, index.html, verificationCode.html 스타일 적용 css
- **web/css/style.css** : Main_Home.jsp 스타일 적용 css
- **web/Find_pw.html** : 사용자 비밀번호를 찾는 인터페이스 html
- **web/Find_pw_Check.jsp** : 비밀번호 찾기를 위한 jsp 파일
- **web/Main_Home.jsp** : 로그인 시 보여지는 메인 페이지 
- **web/Reset_pw.html** : 비밀번호 재설정 페이지
- **web/Reset_pw.jsp** : 비밀번호 재설정을 위한 jsp 파일
- **web/SignUp.html** : 회원가입 페이지
- **web/Signup_Check.jsp** :  회원가입을 확인하는 jsp 파일
- **web/index.html** : 링크 접속 시 보여지는 로그인 페이지 (가장 첫 번째 페이지)
- **web/login_Check.jsp** : 로그인 시도를 위한 jsp 파일
- **web/test.jpg** : 테스트를 위해 입력한 이미지
- **web/verificationCode.html** : 이메일 인증 페이지
- **web/vierifacationCode.jsp** : 이메일 인증을 위한 jsp 파일

### Branch : KTK
- **Image_Captioning_test.py** : 이미지 캡셔닝 모델을 위한 파이썬 코드
- **imagefile_chageCSV.py** : 이미지 파일을 읽은 후 이미지 이름에 있는 키워드를 CSV로 작성 후 이름을 숫자로 변경하는 파이썬 코드

### Branch : SJH
- **index.html** : 저장된 그림책 선택 시 보여지는 UI
- **style.css** : index.html 스타일 적용 css

### Branch : JHJ
- **picturebook.sql** : DB 테이블 생성 sql

### Branch : main
- **한글학습자료1.zip** : 이미지캡셔닝을 위한 한글학습자료 
- **한글학습자료2.zip** : 이미지캡셔닝을 위한 한글학습자료
