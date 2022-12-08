import cv2
import numpy as np
from utils import image_show

image = cv2.imread('./car.jpg')

# Creating out sharpening filter
filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

sharpen_img = cv2.filter2D(image, -1, filter)
cv2.imshow("org image ", image)
cv2.waitKey(0)
image_show(sharpen_img)

'''
입력 영상의 깊이(src, depth())  | 지정 가능한 ddepth 값
CV_8U                            -1/CV_16S/CV_32F/CV64F
CV_16U/CV_16S                    -1/CV_32F/CV_64F
CV_32F                           -1/CV_32F/CV_64F
CV_64F                           -1/CV_64F
'''