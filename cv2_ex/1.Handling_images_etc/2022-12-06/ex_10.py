import cv2
from utils import image_show
import numpy as np

image_path = "./cat.jpg"
image = cv2.imread(image_path)
# image 10x10 픽셀 크기로 변환
image_color_10x10 = cv2.resize(image, (10, 10))
image_shape_info = image_color_10x10.flatten().shape
# print("image_shape_info", image_shape_info) # image_shape_info (300,)
image_show(image_color_10x10)

# image 255x255 픽셀 크기로 변환
image_color_255x255 = cv2.resize(image, (255, 255))
image_color_255x255.flatten()
# image_show(image_color_255x255)

x = np.array([[51, 40], [14, 19], [10, 7]])
x = x.flatten()
# [51, 40, 14, 19, 10, 7]
print(x)
''' 결과
[51 40 14 19 10  7]
'''