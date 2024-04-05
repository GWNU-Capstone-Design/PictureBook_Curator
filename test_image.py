import os
import pandas as pd

# 폴더에 있는 사진 파일 읽어오기
photo_folder = 'C:/Users/xorua/OneDrive/Desktop/Capston/ImageFiletest/'  # 폴더 경로 설정
photo_files = os.listdir(photo_folder)

# 사진 파일 이름과 가상의 키워드 데이터 생성
data = []
for file in photo_files:
    # 사진 파일 이름에서 확장자 제거
    file_name = os.path.splitext(file)[0]
    # 가상의 키워드 생성 (예: 파일 이름에 따라)
    comment = ' '.join(file_name.split('_'))  # 파일 이름을 '_'를 기준으로 나누어 키워드 생성
    data.append({'image_name': file, 'comment': comment})

# 데이터프레임 생성
df = pd.DataFrame(data)

# CSV 파일로 저장
csv_file = 'photo_keywords.csv'  # CSV 파일 이름 설정
df.to_csv(csv_file, index=True, sep='|', index_label='comment_number')  # index=False를 설정하여 인덱스를 CSV 파일에 포함시키지 않음
