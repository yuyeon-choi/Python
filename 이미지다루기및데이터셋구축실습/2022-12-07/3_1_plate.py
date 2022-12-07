import cv2
import matplotlib.pyplot as plt
import numpy as np

def imshow(src, windowName='show', close=True):
    cv2.imshow(windowName, src)
    cv2.waitKey(0)
    if close:
        cv2.destroyAllWindows()

img_ori = cv2.imread('./car1.png')    # cv2.IMREAD_GRAYSCALE 를 쓰면 색상이 없어서 채널에 대한 정보가없어서 오류남
# rgb_img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

# 픽셀 R, G, B 필드가 있는데.. 확인하고 다시 설명해주신다고함!
rgb_img = img_ori[:,:,::-1] # [R, G, B] 이거고  [R, G, B, ::-1] 하면 [ B, G, R]
rgb_img[:,:,0] = 0
rgb_img[:,:,1] = 0
# rgb_img[:,:,2] = 0
# imshow(rgb_img)

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

# convolution Gaussian Filter
img_blurred = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=0)

img_blur_thresh = cv2.adaptiveThreshold(            # adaptiveThreshold 에 대해 다시 찾아보기 이해안됨
    img_blurred, 
    maxValue=255.0, 
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # 가우시안 커널을 이용해 이진화 가우시안값에서 C값을 뺀값을 가우시안 값을 얻어서..? 각각의 윈도우마다,,, ㅠㅠ 
    thresholdType=cv2.THRESH_BINARY_INV, 
    blockSize=19,    # odd over 3
    C=9             # C값은 세부 조정을 하기위해 사용

)

img_thresh = cv2.adaptiveThreshold(
    img_gray, 
    maxValue=255.0, 
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    thresholdType=cv2.THRESH_BINARY_INV, 
    blockSize=19,    # odd over 3
    C=9             # C값은 세부 조정을 하기위해 사용 
)

# plt.figure(figsize=(15,15))
# img_type = ['gray', 'blur', 'orig_thresh', 'blur_thres']
# img_type_array = [img_gray, img_blurred, img_thresh, img_blur_thresh]

# for idx, (name, image) in enumerate(zip(img_type, img_type_array)):
#     plt.subplot(2, 2, idx+1)
#     plt.imshow(image, 'gray')
#     plt.title(name)
# plt.tight_layout()
# plt.show()

contours, _ = cv2.findContours(
    img_blur_thresh, 
    mode=cv2.RETR_LIST, 
    method = cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)
cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(0, 255, 0)) # contourIdx=-1 : 모든 외각선을 표시(?)
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
    cv2.rectangle(temp_result, pt1=(x,y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
    cv2.imshow('temp_result',temp_result)
    cv2.waitKey(0)
imshow(temp_result)
