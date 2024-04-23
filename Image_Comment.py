# 이미지 이름 | 번호 | 키워드
# 이미지 이름을 복사하여 키워드 칸에 저장하는 코드

import os
import pandas as pd

# 폴더에 있는 사진 파일 읽어오기
photo_folder = 'C:/Users/xorua/OneDrive/Desktop/Capston/ImageFiletest/'  # 폴더 경로 설정
photo_files = os.listdir(photo_folder)

# 사진 파일 이름과 가상의 키워드 데이터 생성
data = []
for i, file in enumerate(photo_files):
    # 가상의 키워드 생성 (예: 파일 이름에 따라)
    comment = ' '.join(os.path.splitext(file)[0].split('_'))  # 파일 이름을 '_'를 기준으로 나누어 키워드 생성
    # 데이터 딕셔너리에 이미지 파일 이름, 인덱스 번호, 그리고 해당하는 가상의 키워드를 추가
    data.append({'image_name': file, 'comment_number': i, 'comment': comment})

# 데이터프레임 생성
df = pd.DataFrame(data)

# CSV 파일로 저장
csv_file = 'photo_keywords1.csv'  # CSV 파일 이름 설정
# index=False로 설정하여 DataFrame의 인덱스를 CSV 파일에 포함시키지 않음
# sep='|'로 설정하여 '|'를 구분자로 사용함
df.to_csv(csv_file, index=False, sep='|')

# 1. os.listdir() 함수를 사용하여 지정된 폴더에서 사진 파일 목록을 가져옵니다.
# 2. enumerate() 함수를 사용하여 파일 목록의 각 항목에 대한 인덱스와 값을 반복적으로 가져옵니다.
# 3. 각 사진 파일의 이름을 기반으로 가상의 키워드를 생성합니다. 여기서는 파일 이름을 '_'로 분리하고 공백으로 연결하여 가상의 키워드를 생성합니다.
# 4. 생성된 이미지 파일 이름, 인덱스 번호, 그리고 해당하는 가상의 키워드를 딕셔너리로 만들고, 이를 리스트에 추가합니다.
# 5. 리스트를 사용하여 DataFrame을 생성합니다.
# 6. 생성된 DataFrame을 CSV 파일로 저장합니다. 여기서는 '|'를 구분자로 사용하고, DataFrame의 인덱스를 CSV 파일에 포함시키지 않습니다.
