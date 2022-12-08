import cv2
import numpy as np

# 1. 얼굴 및 눈 감지를 위해 OpenCV Haar 캐스케이드 구성
# creating face_cascade and eye_cascade objects
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

# 얼굴이미지 가져오기
img = cv2.imread('./face.png')

print(img.shape)
cv2.imshow('image show', img)
cv2.waitKey(0)

# >> 케스케이드 경우는 그레이 스케일 이미지에서만 작동
# Converting the image into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Creating variable faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4) # 4 = 박스 4 개나오게 하는것

# Defining and drawing the rectangles around the face
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
    # 좌표에대한 데이터, (255, 0, 255) 색상, 2 선의굵기
cv2.imshow('face', img)
cv2.waitKey(0)

# 관심영역 만들기
roi_gray = gray[y:(y+h), x:(x+w)]
roi_color = img[y:(y+h), x:(x+w)]

# Creating variable eyes
# 바운딩박스안에있는 얼굴에만 gray_scale 줌

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

if eye_1[0] < eye_2[0]:
    left_eye = eye_1
    right_eye = eye_2
else:
    left_eye = eye_2
    right_eye = eye_1

# 5. 두 눈의 중심점 사이에 선을 긋는다. 그 전에 직사각형 중심점의 좌표를 계산해야 합니다.
# Calculating coordinates of a central points of the rectangles

left_eye_center = (int(left_eye[0] + (left_eye[2]/2)),
                   int(left_eye[1] + (left_eye[3] / 2)))
print(left_eye_center)
left_eye_center_x = left_eye_center[0]
left_eye_center_y = left_eye_center[1]

right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3] / 2)))

right_eye_cetner_x = right_eye_center[0]
right_eye_cetner_y = right_eye_center[1]

cv2.circle(roi_color, left_eye_center, 5, (255, 0, 0), -1)
cv2.circle(roi_color, right_eye_center, 5, (255, 0, 0), -1)
cv2.line(roi_color, right_eye_center, left_eye_center, (0, 200, 200), 1)
cv2.imshow('face', img)
cv2.waitKey(0)

if left_eye_center_y > right_eye_cetner_y:
    A = (right_eye_cetner_x, left_eye_center_y)
    direction = - 1 # 정수 -1 은 이미지가 시계방향으로 회전함을 나타냅니다.
else:
    A = (left_eye_center_x, right_eye_cetner_y)
    direction = 1 # 정수 1 은 이미지가 시계 반대 방향으로 회전함을 나타냅니다.
cv2.circle(roi_color, A, 5, (255, 0, 0), -1)
cv2.line(roi_color, left_eye_center, A, (0, 200, 200), 1)
cv2.line(roi_color, right_eye_center, A, (0, 200, 200), 1)

cv2.imshow("face", img)
cv2.waitKey(0)

# 각도 구하기
# np.arctan = 함수 단위는 라디안 단위
# 라디안 단위 -> 각도 : (theta * 180) / np.pi
delta_x = right_eye_cetner_x - left_eye_center_x
delta_y = right_eye_cetner_y - left_eye_center_y
angle = np.arctan(delta_y / delta_x)
angle = (angle * 180) / np.pi
print(angle)
'''
(300, 332, 3)
(63, 104)
'''

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
cv2.imshow('face11', rotated)
cv2.waitKey(0)