from torch.utils.data import Dataset
import os
import glob
import cv2

class CustomDataset(Dataset) :
    def __init__(self, path , transform=None):
        ## path -> ./dataset/train/
        self.all_path = glob.glob(os.path.join(path, "*", "*.png")) # * => bird drone 경로 저장
        self.transform = transform

        # label dict
        self.label_dict = {}
        for index , (category) in enumerate(sorted(os.listdir(path))) : #[bird, drone]
            self.label_dict[category] = int(index)


    def __getitem__(self, item):
        # 1. Reading image
        image_file_path = self.all_path[item]
        image = cv2.imread(image_file_path) # resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. class label
        folder_name = image_file_path.split("\\")
        # print(folder_name)
        # exit()
        folder_name = folder_name[1]
        label = self.label_dict[folder_name]

        # 3. Applying transforms on image
        if self.transform is not None :
            image = self.transform(image=image)["image"]

        # 4. return image, label
        return image, label
    def __len__(self):
        return len(self.all_path)


test = CustomDataset("./dataset/train" , transform=None)
for i in test :
    print(i)

# print(folder_name)
# exit() 
# 하면서 코드 해보기