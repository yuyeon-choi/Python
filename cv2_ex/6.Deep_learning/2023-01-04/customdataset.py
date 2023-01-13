import os
import cv2
import glob
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, file_path, transform= None):
        self.file_path = glob.glob(os.path.join(file_path, "*", "*.png"))
        self.transform = transform

        self.label_dict = {}
        label_names = os.listdir(file_path)
        for value, key in enumerate(label_names):
            self.label_dict[key] = value

    def __getitem__(self, index):
        image_path = self.file_path[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        label_name = image_path.split('\\')[1]
        label = self.label_dict[label_name]
        return image, label

    def __len__(self):
        return len(self.file_path)


# test = CustomDataset("./0104/dataset/train")
# for i in test:
#     print(i)