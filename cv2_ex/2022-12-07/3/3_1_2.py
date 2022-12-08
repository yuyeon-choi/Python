import cv2
import numpy as np
import matplotlib.pyplot as plt
from plot import imshow, plot_images

img_ori = cv2.imread('./car1.png')    # cv2.IMREAD_GRAYSCALE 를 쓰면 색상이 없어서 채널에 대한 정보가없어서 오류남

'''
channel : BGR
height : height
width : width
'''

# image size check
height, width, channel = img_ori.shape
print(height, width, channel)

# # Convert Image to grayscale
img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

'''
가우시안 필터를 사용해 주파수를 낮춰주면 노이즈가 줄어듦
가우시안 커널을 통해서 완만하게 만들어주면 각각의 픽셀에 있는 색상의 값이(명암비)
명암대비 차이가 줄어들게되면서 Threshholding을 했을때 노이즈를 많이 줄일수 있음. 
(그래서 가우시안사용하면 사진이 좀 흐릿해짐)
'''

# Convolution Gaussian Filter
img_blurred = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=0)
plot_images([img_ori, img_gray, img_blurred], ['img_ori', 'img_gray', 'img_blurred'])
# imshow(img_blurred) # 흐릿해진걸 확인할 수 있다.
