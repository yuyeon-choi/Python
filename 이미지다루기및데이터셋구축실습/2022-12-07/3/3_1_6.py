import cv2
import numpy as np
import matplotlib.pyplot as plt
from plot import imshow


img_ori = cv2.imread('./car1.png')
img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
img_blurred = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=0)

# thresh는 3_1_3.py 에 설명해둠__________________________
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

# __________________________________________________________
## Select Candidates by Arrangement of Contours
"""
남은 contours 중에 확실하게 번호판을 찾기 위해 기준을 강화한다.
번호판의 특성을 고려했을 때 세울 수 있는 기준은 아래와 같다.

1. 번호판 Contours의 width와 height의 비율은 모두 동일하거나 비슷하다.
2. 번호판 Contours 사이의 간격은 일정하다.
3. 최소 3개 이상 Contours가 인접해 있어야한다. (대한민국 기준)
"""

MAX_DIAG_MULTIPLAYER = 5
MAX_ANGLE_DIFF = 12.0
MAX_AREA_DIFF = 0.5
MAX_WIDTH_DIFF = 0.8
MAX_HEIGHT_DIFF = 0.2
MIN_N_MATCHED = 3

# cnt_recursive = 0
def find_chars(contour_list):
    # global cnt_recursive
    # cnt_recursive += 1
    matched_result_idx = []

    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            
            # 길이 측정
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            # MAX 값보다 작아야한다. (위에 정의한 값(MAX값)보다 차이(DIFF)가 작아야한다.) 
            if distance < diagonal_length1 * MAX_DIAG_MULTIPLAYER \
                    and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                    and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])
        matched_contours_idx.append(d1['idx'])

        # 최소 갯수(MIN_N_MATCHED = 3)를 만족할 때 까지 반복
        # 만약 끝까지 갔는데도 못찾으면 for문 완료
        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        # np.take 를 사용해 unmatched_contour_idx 들어있는 인덱스 정보에서 possible_contours에 정보를 가져온다.
        unmatched_contour = np.take(possible_contours, unmatched_contour_idx) 
        recursive_contour_list = find_chars(unmatched_contour)

        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break
    return matched_result_idx

result_idx = find_chars(possible_contours)

matched_result = []
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for r in matched_result:
    for d in r:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(0, 255, 0),
                      thickness=2)

cv2.imshow("countours box", temp_result)
cv2.waitKey(0)