import os
from glob import glob
import matplotlib.pyplot as plt

def image_show(data_path):
    x_data, y_data = [], []
    # x축 : 이름, y축 : 데이터 수
    for folder_name in os.listdir(data_path):
        x_data.append(folder_name)
        y_data.append(len(os.listdir(os.path.join(data_path, folder_name))))
        print("Data 이름 : ", x_data[-1])
        print("Image 개수 : ", y_data[-1], '\n')

    # plot
    plt.subplots(figsize= (15,4))
    plt.title("Data Information")
    plt.xlabel("No. of images")
    plt.ylabel("Type of Data")

    plt.barh(x_data, y_data, color="maroon", height= 0.5)
    plt.show()

image_show("./0104/data")