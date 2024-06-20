# PictureBook_Curator
강릉원주대학교 컴퓨터공학과 2024년 1학기 캡스톤디자인I 4조

## 📖 프로젝트 소개
그림책 이미지를 사진 찍으면 이미지캡셔닝을 통해 그림을 설명하는 키워드를 추출합니다.

GPTs를 활용하여 이미지 내용을 설명하는 시나리오를 생성합니다.

생성한 시나리오를 TTS를 통해 목소리로 들려주는 프로젝트입니다.

## 🔗 링크


(서버 열릴 시 접근 가능)

## 🎞️영상


https://github.com/GWNU-Capstone-Design/PictureBook_Curator/assets/114921436/73fab9bf-b911-4854-b59e-c975299beb90



## 🛠️실행 과정

![Group 1](https://github.com/GWNU-Capstone-Design/PictureBook_Curator/assets/114921436/d600b704-d832-4924-a8f1-6380dda543f7)


## 🕰️개발 기간
- 2024.03.04.(월) ~ 2024.05.24.(금)
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
- 2024.04.26. 7차 회의: 중간 점검 및 PictureBook Curator 개발 합치기 1차
  	- 이미지캡셔닝: TansorFlow 버전이 낮아 버전이 높은 이미지캡셔닝 모델을 가져와서 수정 중
  	- DB: 개인 서버를 이용하여 구축, Web과 연동하여 데이터 정보 받기 성공
  	- Web: DB와 연동 및 인터페이스 보안, UI 수정, 카메리 기능 넣을 예정
  	- Github: README를 이용하여 프로젝트 소개 글 작성
- 2024.05.03. 8차 회의: PictureBook Curator 개발 합치기 2차
  	- 이미지캡셔닝: 모델을 이용하여 정확도 확인, 어떻게 모델을 적용할지 의논
  	- Web: 기능 추가 및 보안, 모델 적용 방법 찾기, node.js를 이용하여 적용
- 2024.05.08. 9차 회의: PictureBook Curator 개발 합치기 3차
  	- 모델을 가져올 방안 탐색 및 적용 1
  	- AI: 학습데이터 추가 및 정확도 올리기
  	- Web: 메인 사이트에 기능 상호작용, 스토리보드 작성, flask를 이용해 모델 적용
  	- DB: 파일 업로드 시스템 임시 구현
- 2024.05.17. 10차 회의: PictureBook Curator 개발 합치기 4차
  	- 모델을 가져올 방안 탐색 및 적용 2
  	- AI: 학습데이터 추가 및 정확도 올리기, chat GPTs 기능 설정 및 API 개발
  	- Web: 메인 사이트에 기능 상호작용, 스토리보드 작성, flask를 이용해서 모델 적용
  	- DB: 프론트와 연결할 수 있는 엔드포인트 기능
- 2024.05.17. 10차 회의: PictureBook Curator 개발 합치기 5차
  	- AI: 학습데이터 추가 및 정확도 올리기, chat GPTS 기능 설정 및 API 개발
  	- Web: 이미지 뷰 개발하기
  	- DB: DB 테이블 설정 및 엔드 포인터 설정
- 2024.05.24. 10차 회의: PictureBook Curator 개발 합치기 6차
	- Web: 이미지 뷰 화살표 기능 수정
   	- DB, Server
   	- 삭제 기능 추가
	- 이미지 뷰 코드 -> ktk 브랜치 이용
	- 이미지 뷰에 업로드된 사진 넣기
	- 이미지 tts 기능


## 🧑‍💻프로그래머 소개
- **김태겸** : 팀장, AI 학습 및 개발
- **김동찬** : AI 학습 및 개발
- **신용선** : DB 관리 및 설계
- **신지혜** : 웹사이트 및 앱 개발
- **전재영** : 웹사이트 및 앱 개발, TTS와 gpt api 기능 개발 및 연동
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

### Branch : JJY
- **flask/app.py** : 이미지 캡셔닝, 사진업로드, gpt, tts 까지의 모든 기능을 통합한 flask 서버 파일
- **flask/api.py** : 사진을 업로드하면 캡셔닝, gpt, tts의 과정을 거쳐 파일을 제공해주는 api

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
