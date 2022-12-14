import cv2
import numpy as np

# ex-01 -----------------------------------------------------------------------------
img_rectangle = np.ones((400, 400), dtype='uint8')
cv2.rectangle(img_rectangle, (50, 50), (300, 300), (255, 255, 255), -1)
# cv2.imshow("image show", img_rectangle)
# cv2.waitKey(0)

# ex-02 -----------------------------------------------------------------------------
img_circle = np.ones((400, 400), dtype='uint8')
cv2.circle(img_circle, (300, 300), 70, (255, 255, 255), -1)
# cv2.imshow("image show", img_circle)
# cv2.waitKey(0)

# ex-03 -----------------------------------------------------------------------------
bitwiseAnd = cv2.bitwise_and(img_rectangle, img_circle)
# cv2.imshow("image bitwiseAnd", bitwiseAnd)
# cv2.waitKey(0)

bitwiseOr = cv2.bitwise_or(img_rectangle, img_circle)
# cv2.imshow("image bitwiseOr", bitwiseOr)
# cv2.waitKey(0)

bitwiseXor = cv2.bitwise_xor(img_rectangle, img_circle)
# cv2.imshow("Xor", bitwiseXor)
# cv2.waitKey(0)

rec_not = cv2.bitwise_not(img_rectangle)
# cv2.imshow("rectangle not ", rec_not)
# cv2.waitKey(0)

circle_not = cv2.bitwise_not(img_circle)
# cv2.imshow("circle not ", circle_not)
# cv2.waitKey(0)

# ex-04 -----------------------------------------------------------------------------
mask = np.zeros((683, 1024), dtype='uint8')
cv2.rectangle(mask, (60, 50), (280, 280), (255, 255, 255), -1)
cv2.rectangle(mask, (420, 50), (550, 230), (255, 255, 255), -1)
cv2.rectangle(mask, (750, 50), (920, 280), (255, 255, 255), -1)
# cv2.imshow("...", mask)
# cv2.waitKey(0)

# ex-05 -----------------------------------------------------------------------------
# 마스킹 과제는 흰색대신 이미지를 넣어주시면 됩니다. (원하는 이미지 혹은 얼굴이미지)
img_rectangle = np.ones((600, 1000, 3), dtype='uint8')

img1_path = cv2.imread('./car.jpg')
img2_path = cv2.imread('./test.jpg')
img3_path = cv2.imread('./Billiards.png')

img1 = cv2.resize(img1_path, (250, 250))
img2 = cv2.resize(img2_path, (200, 200))
img3 = cv2.resize(img3_path, (150, 150))

x1_offset, y1_offset = 60, 50
x2_offset, y2_offset = 420, 50
x3_offset, y3_offset = 750, 50

x1_end = x1_offset + img1.shape[0]
y1_end = y1_offset + img1.shape[1]
x2_end = x2_offset + img2.shape[0]
y2_end = y2_offset + img2.shape[1]
x3_end = x3_offset + img3.shape[0]
y3_end = y3_offset + img3.shape[1]

img_rectangle[y1_offset:y1_end, x1_offset:x1_end] = img1
img_rectangle[y2_offset:y2_end, x2_offset:x2_end] = img2
img_rectangle[y3_offset:y3_end, x3_offset:x3_end] = img3
cv2.imshow("show", img_rectangle)
cv2.waitKey(0)