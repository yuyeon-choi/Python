import os
import cv2
import glob
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, file_path, transform= None):
        self.file_path = glob.glob(os.path.join(file_path, "*", "*.jpg"))
        self.transform = transform

        # label 딕셔너리 생성
        self.label_dict = {}
        label_names = os.listdir(file_path)
        for value, key in enumerate(label_names):
            self.label_dict[key] = int(value)

    def __getitem__(self, index):
        # reading image
        image_path = self.file_path[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # class label
        label_name = image_path.split('\\')[1] # ABBOTTS BABBLER ~ YELLOW HEADED BLACKBIRD
        label = self.label_dict[label_name] # 0 ~ 450
        
        # Applying transforms on image
        if self.transform is not None:
            image = self.transform(image= image)["image"]

        return image, label

    def __len__(self):
        return len(self.file_path)

# test = CustomDataset("./0105/dataset/train")
# for i in test:
#     print(i)