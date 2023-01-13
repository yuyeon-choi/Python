import os
from glob import glob
import cv2
from sklearn.model_selection import train_test_split

def split_data(data_path):
    for each_path in os.listdir(data_path):
        total_path = os.path.join(data_path, each_path)    
        images = glob(os.path.join(total_path, "*.JPG"))
        
        train_set, val_set = train_test_split(images, test_size= 0.2, random_state= 77)
        val_set, test_set  = train_test_split(val_set, test_size= 0.5, random_state= 77)
        
        save_in_folder(train_set, mode = "train")
        save_in_folder(val_set  , mode = "val")
        save_in_folder(test_set , mode = "test")

        # print("\n-------" ,each_path, "-------")
        # print("No. of Total Image: ", len(images))
        # print("No. of Train Image: ", len(train_set))
        # print("No. of Val Image: "  , len(val_set))
        # print("No. of Test Image: " , len(test_set))

def save_in_folder(data, mode):
    for path in data:
        folder_name = path.split('\\')[-2]  # Acadia
        image_name  = path.split('\\')[-1]  # IMG_6346.JPG
        image_folder = os.path.join(data_path.replace("dataset", f"data\{mode}"), folder_name)
        os.makedirs(image_folder, exist_ok=True)               # C:\0103\data\train\Acacia 폴더 생성
        file_location = os.path.join(image_folder, image_name) # C:\0103\data\train\Acacia\IMG_6346.JPG 이미지 path
        
        # 이미지 읽어와서 저장
        image = cv2.imread(path)
        cv2.imwrite(file_location.replace(".JPG", ".png"), image)

data_path = "C:\\0103\\dataset"
split_data(data_path)