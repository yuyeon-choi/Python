import cv2
import numpy as np
import matplotlib.pyplot as plt
from plot import imshow


img_ori = cv2.imread('./car1.png')
img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
img_blurred = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=0)

# thresh는 3_1_3.py 에 설명해둠
img_blur_thresh = cv2.adaptiveThreshold(            
    img_blurred, 
    maxValue=255.0,                                 
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  
    thresholdType=cv2.THRESH_BINARY_INV,            
    blockSize=19,   
    C=9             
)

img_thresh = cv2.adaptiveThreshold(
    img_gray, 
    maxValue=255.0, 
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    thresholdType=cv2.THRESH_BINARY_INV, 
    blockSize=19,    # odd over 3
    C=9             
)

'''
channel : BGR
height : height
width : width
'''
height, width, channel = img_ori.shape
# print(height, width, channel)
# __________________________________________________________
contours, _ = cv2.findContours(
    img_blur_thresh, 
    mode=cv2.RETR_LIST,                 # 외곽선 검출 모드
    method = cv2.CHAIN_APPROX_SIMPLE    # 외곽선 근사화 방법.
)

# __________________________________________________________
contours_dict = []

for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    contours_dict.append(
        {
            'contour':contour, 
            'x':x, 
            'y':y, 
            'w':w,
            'h':h, 
            'cx':x+(w/2), 
            'cy':y+(h/2)
        }        
    )
# __________________________________________________________
MIN_AREA = 80
MIN_WIDTH, MIN_HEIGHT = 2, 8
MIN_RATIO, MAX_RATIO = 0.25, 0.8

possible_contours = []

cnt = 0
for d in contours_dict:
    area = d['w'] * d['h']
    ratio = d['w'] / d['h']

    if area > MIN_AREA \
            and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for d in possible_contours:
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']),  # pt1, pt2 안적어도된다.
                  color=(0, 255, 0), thickness=1)

imshow(temp_result, 'temp_result')

"""
위 사진은 추려낸 contours들이다.
번호판 위치에 contours들이 선별된 걸 볼 수 있지만
전혀 관련 없는 영역의 contours들도 저장되었다.
이제 더 기준을 강화하여 번호판 글자들을 찾아야한다.
"""

