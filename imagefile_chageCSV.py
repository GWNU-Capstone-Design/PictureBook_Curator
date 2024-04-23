import os
import pandas as pd
import shutil  # 파일 복사를 위해 추가

# 폴더에 있는 사진 파일 읽어오기
photo_folder = 'C:/Users/xorua/OneDrive/Desktop/Capston/imageF/main_image/'  # 폴더 경로 설정
photo_files = os.listdir(photo_folder)
photo_chage = 'C:/Users/xorua/OneDrive/Desktop/Capston/imageF/kor_image'

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
    
    # 변경된 이름으로 이미지 저장
    original_file_path = os.path.join(photo_folder, file)  # 원본 파일 경로
    new_file_path = os.path.join(photo_chage, new_filename)  # 새 파일 경로
    shutil.copy2(original_file_path, new_file_path)  # 원본 파일을 새 파일 이름으로 복사

# 데이터프레임 생성
df = pd.DataFrame(data)

# CSV 파일로 저장
csv_file = 'photo_keywords_MainKor.csv'  # CSV 파일 이름 설정
df.to_csv(csv_file, index=False, sep='|')
