# 같은 의 이미지 블렌딩 실험
import cv2
import matplotlib.pyplot as plt
import numpy as np

large_img = cv2.imread('./cv2_ex/2022-12-08/ex_image.png')
watermark = cv2.imread('./cv2_ex/2022-12-08/ex_image_logo.png')

print("large_image size >> ", large_img.shape)
print("watermakr image size >> ", watermark.shape)

img1 = cv2.resize(large_img,(800,600))
img2 = cv2.resize(watermark,(800,600))

print("img1 size >> ", img1.shape)
print("img2 size >> ", img2.shape)

'''
large_image size >>  (683, 1024, 3)   
watermakr image size >>  (480, 640, 3)
img1 size >>  (600, 800, 3)
img2 size >>  (600, 800, 3)
'''

# 혼합 진행
# # 베이스 5:5
# blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

# 9:1
# blended = cv2.addWeighted(img1, 9, img2, 1, 0)

# 1로 설정
blended = cv2.addWeighted(img1, 1, img2, 1, 0)
cv2.imshow("image show", blended)
cv2.waitKey(0)
# cv2.imshow("image large", large_img)
# cv2.imshow("watermark", watermark)
# cv2.waitKey(0)

# 195p 논리연산자 참조