# 가우시안 블러
import cv2
from utils import image_show

image_path = "./cat.jpg"

# 이미지 읽기
image = cv2.imread(image_path)

image_g_blury = cv2.GaussianBlur(image, (15, 15), 0)    # 짝수 넣으면 오류남 ()
image_show(image_g_blury)

''' Tip
폴더에서 이미지를 다른 폴더로 옮길때 아래의 명령어를 많이 사용함
copy, shift, move
'''