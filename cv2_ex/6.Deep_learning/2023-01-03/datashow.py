import os
import numpy
from glob import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def data_size_show(data_dir):
    x_plt = []
    y_plt = []

    for directory in os.listdir(data_dir):
        x_plt.append(directory)
        y_plt.append(len(os.listdir(os.path.join(data_dir, directory))))

    # creating the bar plot
    flg, ax = plt.subplots(figsize= (10,10))
    plt.barh(x_plt, y_plt, color="maroon")

    # remove x, y Ticks
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    # add padding between axes and labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)

    # show top values 
    ax.invert_yaxis()

    plt.ylabel('Bark Type')
    plt.xlabel('No. of images')
    plt.title('Bark Texture Dataset')
    plt.show()


data_dir = "C:\\0103\\dataset"
data_size_show(data_dir)