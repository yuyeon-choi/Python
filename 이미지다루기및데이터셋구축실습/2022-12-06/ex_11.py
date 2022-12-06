# 평균색 특성 인코딩
import cv2
import numpy as np

image_path = "./cat.jpg"
image = cv2.imread(image_path)
channels = cv2.mean(image)
print("Channels >> ", channels)
observation = np.array([(channels[2], channels[1], channels[0])])
print(observation)
''' 결과
Channels >>  (205.8482456140351, 205.4207560568087, 207.63455722639935, 0.0)
[[207.63455723 205.42075606 205.84824561]]
'''