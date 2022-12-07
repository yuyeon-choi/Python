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

