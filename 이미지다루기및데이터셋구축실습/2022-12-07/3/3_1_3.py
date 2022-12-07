import cv2
import numpy as np
import matplotlib.pyplot as plt
from plot import imshow, plot_images

img_ori = cv2.imread('./car1.png')    # cv2.IMREAD_GRAYSCALE 를 쓰면 색상이 없어서 채널에 대한 정보가없어서 오류남
img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

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

img_blurred = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=0)


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

## -> Gaussian Blur 비적용 / 적용 비교
"""
Thresholding 적용을 보았으니 가우시안 블러를 사용하는 이유를 알기위해
적용했을 때와 적용하지 않았을 때를 출력해본다.
"""
plt.figure(figsize=(10, 10))
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

# #---------------------------------------------------------------------------------
# # 출력방법2
# plt.figure(figsize=(15, 15))
# img_type = ['orig', 'blur', 'orig_thres', 'blur_thres']
# img_type_array = [img_gray, img_blurred, img_thresh, img_blur_thresh]
# for i in range(1, 9, 2):
#     plt.subplot(2, 4, i)
#     plt.title(f'{img_type[(i - 1) // 2]}_img')
#     plt.imshow(img_type_array[(i - 1) // 2], 'gray')
#     plt.subplot(2, 4, i + 1)
#     # print((i-1)//2)
#     plt.title(f'{img_type[(i - 1) // 2]}_hist')
#     plt.hist(img_type_array[(i - 1) // 2].ravel(), 256)
# plt.tight_layout()
# plt.show()

# cv2.imshow('img_blurred', img_blurred)
# cv2.imshow("img_thresh", img_thresh)
# cv2.imshow("img_blur_thresh", img_blur_thresh)
# cv2.waitKey(0)

# # 언뜻보기엔 큰 차이를 못느낄 수 있지만 번호판 밑부분을 보면 좀 더 검은색 부분이 많아졌다.