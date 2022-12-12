import pydicom
import matplotlib.pyplot as plt
import numpy as np
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

# pip install pydicom

window_cnet = -600
window_width = 1600

# DICOM 파일 읽어오는 함수 -> pydicom.read_file() [더 직관적인 이 함수를 많이 사용!], pydicom.dcmread()
slice = pydicom.read_file("./ID_0000_AGE_0060_CONTRAST_1_CT.dcm")
# print(slice)

s = int(slice.RescaleSlope)
b = int(slice.RescaleIntercept)
image = s * slice.pixel_array + b

plt.subplot(1, 3, 1)
plt.title("DICOM -> Array")
plt.imshow(image, cmap='gray')

slice.WindowCenter = window_cnet
slice.WindowWidth = window_width
image = apply_modality_lut(image, slice)
image2 = apply_voi_lut(image, slice)
plt.subplot(1,3,2)
plt.title("apply_voi_lut")
plt.imshow(image2, cmap='gray')
# plt.show()

# normalization
image3 = np.clip(image, window_cnet-(window_width/2),
                  window_cnet + (window_width/2))

plt.subplot(1, 3, 3)
plt.title("normalization")
plt.imshow(image3, cmap='gray')
plt.show()

