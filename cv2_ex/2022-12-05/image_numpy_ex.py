import numpy as np
import cv2
# pip install opencv-python==4.5.5.62 최신버전일수록 오류가 있음 이 버전을 추천!

# 이미지 경로 
x = cv2.imread("./cat.jpg", 0)   # 흑백 이미지
y = cv2.imread("./cat.jpg", 1)   # 컬러 이미지

# img = cv2.resize(x, (200, 200)) 사이즈 변경
cv2.imshow('cat image show gray ', x)
cv2.imshow('cat image show color ', y)
# cv2.waitKey(0)

# 여러개 파일 save .npz
np.savez("./image.npz", array1=x, array2=y)

# 압축 방법 C:\Users\yuyeon\Documents\GitHub\Python 에서 확인(936KB -> 433KB)
np.savez_compressed("./image_compressed.npz", array1=x, array2=y)

# npz 데이터 로드
data = np.load("./image_compressed.npz")

result1 = data['array1']
result2 = data['array2']

cv2.imshow("result01", result1)
cv2.waitKey(0)      # 이미지일때는 0, 비디오일때는 1


