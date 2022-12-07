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
"""
Contours란 동일한 색 또는 동일한 강도를 가지고 있는 영역의 경계선을 연결한 선이다.

findContours()는 이런 Conturs들을 찾는 opencv 메소드이다.
위 메소드는 검은색 바탕에서 흰색 대상을 찾는다.
그래서 4번째 단계에서 Thresholding을 해주고 가우시안 블러를 적용해준 것이다.

그런데 공식문서에는 findCountours의 리턴 값으로
image, contours, hierachy 이렇게 3개가 나온다고 나와있지만
현재 첫번째 리턴 값인 image가 사라진 듯하다.
그래서 contours와 로 리턴을 받았다. hierachy는 쓸 일이 없어 로 받음

사진의 윤곽선을 모두 딴 후 opencv의 drawContours() 메소드로
원본사진이랑 크기가 같은 temp_result란 변수에 그려보았다
"""

"""
findContours : 안에 있는 값들 중에서 이어지는 것들의 뭉치. 
               컨투어 정보(x, y)와 구조 정보(내부의 값들이 있는데 이전의 값과 다음의 값이 연관관계가 있느냐에 대한 계층구조를 나타냄(이번 실습에서는 사용안하므로 신경 안써도 된다.))
"""
contours, _ = cv2.findContours(
    img_blur_thresh, 
    mode=cv2.RETR_LIST,                 # 외곽선 검출 모드
    method = cv2.CHAIN_APPROX_SIMPLE    # 외곽선 근사화 방법.
)

# color: 외곽선 색상 • thickness: 외곽선 두께. thinkness < 0이면 내부를 채운다.
temp_result = np.zeros((height, width, channel), dtype=np.uint8)
cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(0, 255, 0)) # contourIdx=-1 : 외각선 인덱스. -1 값을 주면 모든 컨투어들을 다 그림
imshow(temp_result, 'temp_Result')


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
    #rectangle : 왼쪽 상단의 좌표와 오른쪽 하단의 좌표를 넣어줘야함 (따라서 pt2를 보면 x,y 좌표에 너비와 높이를 더해줌)
    cv2.rectangle(temp_result, pt1=(x,y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2) 
    # cv2.imshow('temp_result',temp_result)
    # cv2.waitKey(0)
# imshow(temp_result)