import cv2


def image_show(image):
    cv2.imshow("show", image)
    cv2.waitKey(0)


image_path = "./cat.jpg"

# 이미지 읽기
image = cv2.imread(image_path)

# 이미지 크롭 [시작 : 끝 : 단계]
image_crop = image[10:, :500]

# 저장코드 추가 png 파일 저장

image_show(image_crop)