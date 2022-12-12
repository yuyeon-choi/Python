import cv2
import matplotlib.pyplot as plt
import numpy as np

def imshow(src, windowName='show', close=True):
    cv2.imshow(windowName, src)
    cv2.waitKey(0)
    if close:
        cv2.destroyAllWindows()

# 이 코드는 자세히 볼 필요없다.!
def plot_images(image_list:list, title_list:list):
    img_cnt = len(image_list)
    plt.figure(figsize=(15, 15))
    for i in range(1, img_cnt+1):
        plt.subplot(1, img_cnt+1, i)
        try:
            h, w, c = image_list[i-1].shape
            if c == 3:
                plt.imshow(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
            else:
                continue
        except:
            plt.imshow(img_gray, 'gray')
        finally:
            plt.title(title_list[i-1])
    plt.show()



img_ori = cv2.imread('./car1.png')    # cv2.IMREAD_GRAYSCALE 를 쓰면 색상이 없어서 채널에 대한 정보가없어서 오류남
rgb_img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

# 필기 참조
rgb_img == img_ori[:,:,::-1] # [R, G, B] 이거고  [R, G, B, ::-1] 하면 [ B, G, R]
rgb_img[:,:,0] = 0
rgb_img[:,:,1] = 0
# rgb_img[:,:,2] = 0
imshow(rgb_img)

# imshow(img_ori, 'show', False) 

'''
channel : BGR
height : height
width : width
'''
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

# Adaptive Thresholding
"""
Thresholding을 해주기 전에 가우시안 블러를 해주는 것이 번호판을 더 잘 찾게 만들어 줄 수 있다.
가우시안 블러는 사진의 노이즈를 없애는 작업이다.
가우시안 블러를 적용해야하는 이유는 아래 4-1에서 설명한다.

그럼 먼저 Thresholding을 살펴보자.
Thresholding 이란 지정한 threshold 값을 기준으로 정하고
이보다 낮은 값은 0, 높은 값은 255로 변환한다. 즉 흑과 백으로만 사진을 구성하는 것이다.

이걸 해주는 이유는 5번째 단계에서 Contours를 찾으려면 검은색 배경에 흰색 바탕이어야 한다.
또 육안으로 보기에도 객체를 더 뚜렷하게 볼 수 있다.
"""

'''
Threshold : 지정한 threshold 값을 기준으로 정하고
이보다 낮은 값은 0, 높은 값은 255로 변환한다. 즉 흑과 백으로만 사진을 구성하는 것이다.
Threshold는 그림자와 명암 등등을 고려하지 못한다. 따라서 원하는대로 이진화가 안될 가능성이 높아진다.
adaptiveThreshold : 각각의 윈도우마다 평균값(mean)에서 특정값을 뺀값의 Threshold를 생성한다.
이를 adaptiveThreshold 라 하고 주변 그림자와 음영 등등 주변을 고려하여 이진화함으로 Threshold의 단점을 보완해준다.
'''

img_blur_thresh = cv2.adaptiveThreshold(            
    img_blurred, 
    maxValue=255.0,                                 # 255가 max값으로 이진화를 한다. 
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 블록 평균 계산 방법 지정. 가우시안 연산을 통해서 19개의 블록을 확인한 값에서 C값을 뺀것을 threshold값으로 지정한다.
    thresholdType=cv2.THRESH_BINARY_INV,            # 이진화 반전. 생성된 기준값보다 높으면 255, 아니면 0으로 만드는데 이것을 반전시킨다.
    blockSize=19,    # 블록 크기. 3 이상의 홀수
    C=9             # 블록 내 평균값 또는 블록 내 가중 평균값에서 뺄 값. (x, y) 픽셀의 임계값으로 𝑇(𝑥, 𝑦) = 𝜇(𝑥, 𝑦 )− 𝐶 를 사용/ C값은 세부 조정을 하기위해 사용
)

img_thresh = cv2.adaptiveThreshold(
    img_gray, 
    maxValue=255.0, 
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    thresholdType=cv2.THRESH_BINARY_INV, 
    blockSize=19,    # odd over 3
    C=9             
)

plt.figure(figsize=(15,15))
img_type = ['gray', 'blur', 'orig_thresh', 'blur_thresh']
img_type_array = [img_gray, img_blurred, img_thresh, img_blur_thresh]

for idx, (name, image) in enumerate(zip(img_type, img_type_array)):
    plt.subplot(2, 2, idx+1)
    plt.imshow(image, 'gray')
    plt.title(name)
plt.tight_layout()
plt.show()
'''
결과창 사진은 아래를 나타냄
ㅁㅁ = gray         blur(GAUSSIAN)
ㅁㅁ = gray_thresh  blur(GAUSSIAN)_thres
=> GAUSSIAN 처리를 한 사진이 더 부드럽게 나옴.
'''
# # ------------------------------------------------------------------------
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
    cv2.imshow('temp_result',temp_result) 
    cv2.waitKey(0)
imshow(temp_result)

# ____________________________________________________________
# 아래값은 변경해가면서 최적의 값을 찾아내면 된다. (1:51:00)
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
                  color=(255, 255, 255), thickness=2)

imshow(temp_result, 'temp_result')
# cv2.waitKey(0)
# cv2.destroyAllWindows()
"""
위 사진은 추려낸 contours들이다.
번호판 위치에 contours들이 선별된 걸 볼 수 있지만
전혀 관련 없는 영역의 contours들도 저장되었다.
이제 더 기준을 강화하여 번호판 글자들을 찾아야한다.
"""

#____________________________________________________________

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

#____________________________________________________________
# [03:05:36]
### Rotate plate image 
PLATE_WIDTH_PADDING = 1.3  
PLATE_HEIGHT_PADDING = 1.5  
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

plate_imgs = []
plate_infos = []

for i, matched_chars in enumerate(matched_result):
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

    # 박스들을 x 값의 센터를 기준으로 정렬함. 처음 박스의 센터에서 마지막 박스의 센터까지의 길이를 통해 구하겠다.
    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h']

    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
    triangle_hypotenus = np.linalg.norm(
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )

    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

    img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))

    img_cropped = cv2.getRectSubPix(
        img_rotated,
        patchSize=(int(plate_width), int(plate_height)),
        center=(int(plate_cx), int(plate_cy))
    )

    if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
        continue

    x = int(plate_cx - plate_width / 2)
    y = int(plate_cy - plate_height / 2)
    w = int(plate_width)
    h = int(plate_height)

    # num_idx = 0
    for num_idx, sorted_char in enumerate(sorted_chars):
        number_crop = cv2.getRectSubPix(
            img_rotated,
            patchSize=(int(sorted_char['w']), int(sorted_char['h'])),
            center=(int(sorted_char['cx']), int(sorted_char['cy']))
        )
        plt.subplot(1, len(sorted_chars), num_idx+1)
        plt.imshow(number_crop, 'gray')
    plt.show()    

    img_out = img_ori.copy()
    cv2.rectangle(img_out, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
    cv2.imshow("test", img_cropped)
    cv2.imshow("orig", img_out)
    cv2.waitKey(0)