'''
opening : erosion -> dilation (to delete dot noise)
'''
#../Bil~.png  =>  .(1개) : 지금 위치 / ..(2개) : 부모 위치

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./Billiards.png', cv2.IMREAD_GRAYSCALE)

_, mask = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)

# datatype : int, float
kernel = np.ones((3, 3), np.uint8)

N = 5 
idx = 1
for i in range(1, N + 1):
    erosion = cv2.erode(mask, kernel, iterations=1)
    opening = cv2.dilate(erosion, kernel, iterations=1)
    f_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    plt.figure(figsize=(15, 15))
    plt.subplot(N, 2, idx)
    idx += 1
    plt.imshow(opening, 'gray')
    plt.title(f' {i} opening')

    
    plt.subplot(N, 2, idx)
    plt.imshow(f_opening, 'gray')
    plt.title('function opening')
plt.show()