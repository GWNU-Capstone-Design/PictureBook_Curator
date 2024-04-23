import os
import pandas as pd

# 폴더에 있는 사진 파일 읽어오기
photo_folder = 'C:/Users/xorua/OneDrive/Desktop/Capston/ImageFiletest_Kor/'  # 폴더 경로 설정
photo_files = os.listdir(photo_folder)

# 사진 파일 이름을 '_'를 기준으로 나누어 키워드 생성하고 파일 이름을 숫자로 변경하여 데이터 생성
data = []
for i, file in enumerate(photo_files):
    # '_'를 기준으로 파일 이름을 분리하여 키워드 생성
    keywords = os.path.splitext(file)[0].split('_')
    comment = ' '.join(keywords)  # 키워드를 공백으로 구분하여 하나의 문자열로 결합

    # 파일 이름을 숫자로 변경
    new_filename = str(i + 1) + '.jpg'

    # 데이터 추가
    data.append({'image_name': new_filename, 'comment_number': 0, 'comment': comment})

# 데이터프레임 생성
df = pd.DataFrame(data)

# CSV 파일로 저장
csv_file = 'photo_keywords_Kor2.csv'  # CSV 파일 이름 설정
df.to_csv(csv_file, index=False, sep='|')
