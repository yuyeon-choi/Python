import os
import cv2
import numpy
from glob import glob
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, path, transform = None):
        self.paths = glob(os.path.join(path, "*", "*.png"))
        self.transform = transform

        self.label_dict = {}
        for index, item in enumerate(os.listdir(path)):
            self.label_dict[os.path.basename(item)] = index

    def __getitem__(self, index):
        image_path = self.paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        # ['C:', '0103', 'data', 'train', 'Acacia', 'IMG_6348.png']
        label_name = image_path.split('\\')[-2]  # Acacia
        label = self.label_dict[label_name]
        # print(image, label)
        return image, label

    def __len__(self):
        return len(self.paths)

# test = CustomDataset("C:\\0103\\data\\train")
# for i in test:
#     pass