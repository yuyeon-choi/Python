import cv2
from utils import image_show

# 이미지 경로
image_path = "./cat.jpg"

# 이미지 이진화
image_grey = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
max_output_value = 255      # 출력값 0 ~ 255 / 출력 칙셀 강도의 최댓값 저장
neighborhood_size = 99      # 보통 99를 많이 사용. 낮추면 검은색 부분이 흐려짐. 홀수만 가능.
subtract_from_mean = 5      # 낮은값을 주는게 좋음 높아지면 많이 날라감.

image_binary = cv2.adaptiveThreshold(image_grey, max_output_value,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, neighborhood_size,  # cv2.THRESH_BINARY_INV 사용하면 반전됨.
                                     subtract_from_mean)

image_show(image_binary)

''' adaptiveThreshold() 함수에는 네 개의 중요한 매개변수가 있다.
• max_output_value : 출력 픽셀 강도의 최댓값 저장
• cv2.ADAPTIVE_THRESH_GAUSSIAN_C : 픽셀의 임곗값을 주변 픽셀 강도의 가중치 합으로 설정. 가중치
는 가우시안 윈도우에 의해 결정
• cv2.ADAPTIVE_THRESH_MEAN_C : 주변 픽셀의 평균을 임곗값으로 설정
'''
