import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ex03_custom_dataset import CustomDataset

import matplotlib.pyplot as plt

def aug_function(mode_flag):
    if mode_flag == "train":
        train_transform = A.Compose([
            A.SmallestMaxSize(max_size= 400),
            A.Resize(width= 256, height =256),
            A.RandomCrop(width= 224, height =224),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit= 0.05, scale_limit= 0.06,
                                rotate_limit=20, p=0.8),
            A.RGBShift(r_shift_limit=17, g_shift_limit=17, b_shift_limit= 17),
            A.RandomBrightnessContrast(p=0.6),
            A.RandomShadow(p=0.6),
            A.RandomFog(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        return train_transform
    elif mode_flag == "valid":
        valid_transform = A.Compose([
            A.SmallestMaxSize(max_size= 400),
            A.Resize(width= 256, height =256),
            A.CenterCrop(width= 224, height =224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        return valid_transform

def visualize_augmentation(dataset, idx = 0, cols= 5):
    dataset = copy.deepcopy(dataset)
    samples = 5
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(
        t, (A.Normalize, ToTensorV2)
    )])
    rows = samples // cols
    figure, ax = plt.subplots(nrows= rows, ncols= cols, figsize=(12,6))

    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    
    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
#     train_aug = aug_function(mode_flag= "train")
#     train_dataset = CustomDataset("./0104/dataset/train", transform= train_aug)
#     visualize_augmentation(train_dataset)