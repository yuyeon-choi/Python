import cv2
# from ex_image03 import image_show

def imshow(image):
    cv2.imshow("show", image)
    cv2.waitKey(0)

image_path = "./cat.jpg"

#이미지 읽기
image = cv2.imread(image_path)

# 이미지 블러
image_blury = cv2.blur(image, (5, 5))
imshow(image_blury)
