import os
import glob
import cv2
from sklearn.model_selection import train_test_split

def data_split(data_path):
    for folder_name in os.listdir(data_path):
        file_path = os.path.join(data_path, folder_name)
        image_list = glob.glob(os.path.join(file_path, '*.jpg'))
        train_set, valid_set = train_test_split(image_list, test_size= 0.1, random_state= 77)

        save_in_folder(train_set, mode= 'train')
        save_in_folder(valid_set, mode= 'valid')

def save_in_folder(path_set, mode):  
    for image_path in path_set:
        new_folder = image_path.split("\\")[0].replace('data', f"dataset/{mode}")
        label_name = image_path.split("\\")[1] # ['./0104/data', 'african-wildcat', 'af (21).jpg']
        image_name = image_path.split("\\")[2].replace('.jpg', '.png')
        full_path = os.path.join(new_folder, label_name, image_name)

        # make folder
        os.makedirs(os.path.join(new_folder, label_name), exist_ok= True)

        # save image
        image = cv2.imread(image_path)
        cv2.imwrite(full_path, image)
      
data_split("./0104/data")