import cv2
import numpy as np
import matplotlib.pyplot as plt
from plot import imshow, plot_images

img_ori = cv2.imread('./car1.png')    # cv2.IMREAD_GRAYSCALE 를 쓰면 색상이 없어서 채널에 대한 정보가없어서 오류남
rgb_img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

# 필기 참조
rgb_img == img_ori[:,:,::-1] # [R, G, B] 이거고  [R, G, B, ::-1] 하면 [ B, G, R]
rgb_img[:,:,0] = 0
rgb_img[:,:,1] = 0
# rgb_img[:,:,2] = 0
imshow(rgb_img)

imshow(img_ori, 'show', False)