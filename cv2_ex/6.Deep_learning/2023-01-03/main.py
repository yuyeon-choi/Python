from ex03_custom_dataset import CustomDataset
from ex05_utils import train

import torch
import torch.nn as nn
import torch.optim as optim
import rexnetv1
from torchvision import models
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from timm.loss import LabelSmoothingCrossEntropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = A.Compose([
    A.SmallestMaxSize(max_size= 256),
    A.Resize(height= 224, width= 224),
    A.RandomShadow(p = 0.5),
    A.RandomFog(p = 0.4),
    A.RandomSnow(p = 0.4),
    A.RandomBrightnessContrast(p = 0.5),
    A.ShiftScaleRotate(shift_limit= 0.05, scale_limit= 0.05,
                    rotate_limit= 15, p = 0.7),
    A.HorizontalFlip(p= 0.5),
    A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
    ToTensorV2()
])
val_transform = A.Compose([
    A.SmallestMaxSize(max_size = 256),
    A.Resize(height=224, width=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# dataset
train_dataset = CustomDataset("C:\\0103\\data\\train", transform= train_transform)
valid_dataset = CustomDataset("C:\\0103\\data\\val"  , transform= val_transform)
test_dataset  = CustomDataset("C:\\0103\\data\\test" , transform= val_transform)

# data loader
train_loader = DataLoader(train_dataset, batch_size=126, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=126, shuffle=False)
test_loader  = DataLoader(test_dataset , batch_size=126, shuffle=False)

# pretrained
model = rexnetv1.ReXNetV1()
model.load_state_dict(torch.load("./0103/rexnetv1_1.0.pth"))
model.output[1] = nn.Conv2d(1280, 50, kernel_size= 1, stride= 1)
model.to(device)

# pretrained no
# model = rexnetv1.ReXNetV1(classes= 50)
# model.to(device)

criterion = LabelSmoothingCrossEntropy()
optimizer = optim.AdamW(model.parameters(), lr= 0.001)
save_dir = "./0103"

num_epoch = 100

train(num_epoch, model, train_loader, valid_loader, criterion, 
    optimizer , save_dir, device)