'''
가우시안 필터 적용 136p
참고자료 (해상도 규격)
https://namu.wiki/w/%ED%95%B4%EC%83%81%EB%8F%84/%EB%AA%A9%EB%A1%9D
'''
# 가우시안 필터 적용
import cv2
import numpy as np
from utils import image_show

img = cv2.imread("./car.jpg", 0)
print(img.shape)
img_rsize = cv2.resize(img, (320, 240))

Gaussian_blurred_1 = np.hstack([
    cv2.GaussianBlur(img_rsize, (3, 3), 0),
    cv2.GaussianBlur(img_rsize, (5, 5), 0),
    cv2.GaussianBlur(img_rsize, (9, 9), 0),
])
image_show(Gaussian_blurred_1)
cv2.imwrite("gaussian_blur.png", Gaussian_blurred_1)    # blur.png 랑 비교해보기
