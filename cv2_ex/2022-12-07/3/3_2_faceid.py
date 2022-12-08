import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 얼굴 및 눈 감지를 위해 OpenCV Haar 캐스케이드 구성
# Creating face_cascade and eye_cascade objects
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

'''
>> class 라는 이름 face_cascade 으로 eye_cascade 객체를 생성 cv2.CascadeClassifier() 했습니다.
   이 두개에 감지된 얼굴과 눈을 저장할 것입니다.
'''

## 2. 얼굴 이미지 데이터 읽기
## Loading the image
img = cv2.imread('./face01.png')
# cv2.imshow("image show", img)
# cv2.waitKey(0)

## 2. 얼굴 이미지 바운딩 박스
## >> 케스케이드 경우는 그레이 스케일 이미지에서만 작동
# Converting the image into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Creating variable faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Defining and drawing the rectangle around the face
for(x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

# cv2.imshow("face", img)
# cv2.waitKey(0)

'''
detectMultiScale() -> 바운딩 박스 좌표를 획득 가능
인자값 : 그레이 이미지, 축소할 이미지 배율 인수, 이웃의 최소 수
'''
'''
 3. 얼굴에 대한 bounding 박스 획득. 눈 감지 해보도록 하겠습니다. 
    이를 위해 먼저 사각형 안에 위치할 두개의 관심 영역을 만들어야합니다. 왜 두개의 관심영역이 필요할까요 ? 
    눈을 감지할 회색조 이미지 첫번째 영역이 필요하고 두 번째 영역은 사각형을 그릴 컬러 이미지가 필요합니다. 
'''
# Creating two regions of interest
roi_gray = gray[y:(y+h), x:(x+w)]
roi_color = img[y:(y+h), x:(x+w)]

# Creating variable eyes
eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
index = 0
# Creating for loop in order to divide one eye from another
for (ex, ey, ew, eh) in eyes:
    if index == 0:
        eye_1 = (ex, ey, ew, eh)
    elif index == 1:
        eye_2 = (ex, ey, ew, eh)
# Drawing rectangles around the eyes
    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0,255), 3)    
    index = index + 1
cv2.imshow("face", img)
cv2.waitKey(0)
# for 루프를 만들어 첫번째 눈과 두 번째 눈의 좌표를 각각 eye_1 변수와 eye_2 변수에 저장

# 4. left_eye 더 right_eye 작은 눈이 우리의 left_eye .
if eye_1[0] < eye_2[0]:
    left_eye = eye_1
    right_eye = eye_2
else:
    left_eye = eye_2
    right_eye = eye_1

# 5. 두 눈의 중심점 사이에 선을 긋는다. 그 전에 직사각형 중심점의 좌표를 계산해야 합니다.
# Calculating coordinates of a central points of the rectangles
left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
left_eye_x = left_eye_center[0]
left_eye_y = left_eye_center[1]

right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
right_eye_x = right_eye_center[0]
right_eye_y = right_eye_center[1]\

cv2.circle(roi_color, left_eye_center, 5, (255, 0, 0), -1)
cv2.circle(roi_color, right_eye_center, 5, (255, 0, 0), -1)
cv2.line(roi_color, right_eye_center, left_eye_center, (0, 200, 200), 3)

cv2.imshow("face", img)
cv2.waitKey(0)
# 인덱스 0 -> x 좌표 / 인덱스 1 -> y 좌표 / 인덱스 2 -> 사각형의 너비 마지막 인덱스 사각형 높이를 나타냅니다.

# 6. 수평선을 그리고 그 선과 눈의 두 중심점을 연결하는 선 사이의 각도를 계산하는 것입니다.
# 최종 목적은 이 각도를 기준으로 이미지를 회전시키는 것입니다

if left_eye_y > right_eye_y:
    A = (right_eye_x, left_eye_y)
    # Integer -1 indicates that the image will rotate in the clockwise direction
    direction = -1
else:
    A = (left_eye_x, right_eye_y)
    # Integer 1 indicates that image will rotate in the counter clockwise direction
    direction = 1

cv2.circle(roi_color, A, 5, (255, 0, 0), -1)

cv2.line(roi_color, right_eye_center, left_eye_center, (0, 200, 200), 3)
cv2.line(roi_color, left_eye_center, A, (0, 200, 200), 3)
cv2.line(roi_color, right_eye_center, A, (0, 200, 200), 3)

cv2.imshow("face", img)
cv2.waitKey(0)

'''
7. 회전할 방향을 지정했다는 점에 유의 !! 왼쪽 눈 y 좌표가 오른쪽 눈 y 좌표 보다 크면 이미지를
시계방향으로 회전합니다. 그렇지 않으면 이미지를 반대 방향으로 회전 합니다.
각도를 계산하려면 먼저 직각 삼각형의 두 변의 길이를 찾아야 합니다. 다음의 공식을 이용
'''
# np.arctan 함수는 라디안 단위로 각도를 반환 한다는점에 유의 결과를 각도로 변환하려면
# 각도 \(\theta(세타) \)에 180을 곱한 다음 \(\pi \)로 나누어야 합니다.
delta_x = right_eye_x - left_eye_x
delta_y = right_eye_y - left_eye_y
angle = np.arctan(delta_y/delta_x)
angle = (angle * 180)/ np.pi

delta_x_1 = right_eye_x + left_eye_x
delta_y_1 = right_eye_y + left_eye_y

# 8. 마지막으로 이미지를 각도 세타 만큼 회전
# Width and height of the image
h, w = img.shape[:2]
# Calculating a center point of the image
# Integer division "//"" ensures that we receive whole number
center = (w // 2, h // 2)
# Defining a matrix M and calling
# cv2.getRotationMatrix2D method
M = cv2.getRotationMatrix2D(center, angle, 1.0)
# Applying the rotation to our image using the cv2.warpAffine method
rotated = cv2.warpAffine(img, M, (w, h))

# 결과 -> -21.80140948635181 도
cv2.imshow("face", rotated)
cv2.waitKey(0)

'''
9. 이제 마지막으로 이미지 크기를 조정해야합니다.
이를 위해 이 이미지에서 눈 사이의 거리를 참조 프레임으로 사용합니다.
하지만 먼저 이 거리를 계산해야 합니다. 우리는 이미 직각 삼각형의 두 변의 길이를 계산했습니다.
따라서 빗변을 나타내는 피타고라스의 정리를 사용하여 두 눈 사이의 거리를 계산할 수 있습니다.
이 코드로 처리하는 다른 모든 사진에 대해 동일한 작업을 수행할 수 있습니다.
그런 다음 이러한 결과의 비율을 계산하고 해당 비율에 따라 이미지를 확장할 수 있습니다.
'''
# dist_1 = np.sqrt((delta_x * delta_x) + delta_y * delta_y)
# dist_2 = np.sqrt((delta_x_1 * delta_x_1) + (delta_y_1 * delta_y_1))

# # calculate the ratio
# ratio = dist_1 / dist_2

# # Defining the width and height
# h = 476
# w = 488
# # Defining aspect ratio of a resized image
# dim = (int(w * ratio), int(h * ratio))
# # We have obtained a new image that we call resized3
# resized = cv2.resize(rotated, dim)

# cv2.imshow("face", resized)
# cv2.waitKey(0)

