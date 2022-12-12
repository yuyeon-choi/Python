import cv2
import matplotlib.pyplot as plt

image_path = "./cat.jpg"

# 이미지 읽기
image = cv2.imread(image_path)
# RGB 타입 변환
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 사이즈 변환
image_50x50 = cv2.resize(image_rgb, (50, 50))

cv2.imwrite("./cat_image_50x50.png", image_50x50)   #jpg보다 png가 고화질/ png가 용량이 더 높긴함
flg, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image_rgb)
ax[0].set_title("Original Image")
ax[1].imshow(image_50x50)
ax[1].set_title("Resize Image")
plt.show()