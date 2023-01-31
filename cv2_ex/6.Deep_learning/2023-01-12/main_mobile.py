import warnings
warnings.filterwarnings('ignore')
import copy
import sys
import os
# import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.models as models
import torch.nn as nn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from ex03_customdataset import CustomDataset
from torch.utils.data import DataLoader
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm



# 0. device setting----------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# 1. augmentation setting
train_aug = A.Compose([
    A.SmallestMaxSize(max_size= 224),
    A.Resize(width= 200, height= 200),
    # A.RandomCrop(width= 180, height= 180),
    A.HorizontalFlip(p=0.6),
    A.VerticalFlip(p=0.6),
    A.ShiftScaleRotate(shift_limit= 0.05, scale_limit= 0.06,
                            rotate_limit=20, p=0.5),
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1),
    A.RandomBrightnessContrast(p= 0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225)),
    ToTensorV2()
    ])

valid_aug = A.Compose([
    A.SmallestMaxSize(max_size= 224),
    A.Resize(width= 200, height= 200),
    # A.CenterCrop(width= 180, height= 180),
    A.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225)),
    ToTensorV2()
    ])

# 2. loading classification dataset-----------------------------------------
train_dataset = CustomDataset("./dataset/train" , transform= train_aug)
valid_dataset = CustomDataset("./dataset/val"   , transform= valid_aug)
test_dataset  = CustomDataset("./dataset/test"  , transform= valid_aug)

'''
# augmentated image check
def visualize_aug(dataset, idx=2, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    # Normalize와 ToTensor를 풀어줘야 함
    dataset.transform = A.Compose([t for t in dataset.transform
                                   if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()

visualize_aug(train_dataset)
exit()
'''
# 3. Data loader-----------------------------------------------------------
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# 4. Model setting----------------------------------------------------------
model = models.mobilenet_v3_large(pretrained=True)
model.classifier[3] = nn.Linear(in_features=1280, out_features=2)
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

# scheduler
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.93)  # lr * gamma

# 5. Hyper parameter setting-----------------------------------------------
epochs = 10
best_val_acc = 0.0
train_step = len(train_loader)
val_step = len(val_loader)
save_path = "mobile_best.pt"
dfForAccuracy = pd.DataFrame(index=list(range(epochs)), columns=['Epoch', "train_acc",
                                                                 'val_acc', "train_loss", "val_loss"])

def train(best_val_acc):
    if os.path.exists(save_path):
        best_val_acc = max(pd.read_csv("./mobile_Accuracy1.csv")["Accuracy"].tolist())
        model.load_state_dict(torch.load(save_path))

    for epoch in range(epochs):
        running_loss = 0
        val_acc = 0
        train_acc = 0
        running_val_loss = 0.0

        model.train()
        # 프로세스 진행바 생성
        train_bar = tqdm(train_loader, file=sys.stdout, colour='green')
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            exp_lr_scheduler.step()

            optimizer.zero_grad()
            train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            train_bar.desc = f"train epoch [{epoch+1}/{epochs}],  loss >> {loss.data:.3f}"

        # 평가모드로 전환
        model.eval()
        with torch.no_grad():  # train이 아니라서 미분필요 x, loss도 필요 x
            valid_bar = tqdm(val_loader, file=sys.stdout, colour='red')
            for data in valid_bar:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                predicted_outputs = model(images)
                val_loss = criterion(predicted_outputs, labels)
                running_val_loss += val_loss.item()

                val_acc += (torch.argmax(predicted_outputs, dim=1) == labels).sum().item()

        val_accuracy = val_acc / len(valid_dataset) * 100
        train_accuracy = train_acc / len(train_dataset) * 100
        train_loss = running_loss / train_step
        valid_loss = running_val_loss / val_step

        dfForAccuracy.loc[epoch, "Epoch"] = epoch + 1
        dfForAccuracy.loc[epoch,"train_acc"] = round(train_accuracy, 3)
        dfForAccuracy.loc[epoch, "val_acc"] = round(val_accuracy, 3)
        dfForAccuracy.loc[epoch, 'train_loss'] = round(train_loss, 4)
        dfForAccuracy.loc[epoch, 'val_loss'] = round(valid_loss, 4)


        print(f"epoch [{epoch+1}/{epochs}]    train_loss: {(running_loss/train_step):.4f}"
              f"    train acc: {train_accuracy:.3f}%    val acc: {val_accuracy:.3f}%")

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), save_path)

        if epoch == epochs -1:
            dfForAccuracy.to_csv('./mobile_Accuracy1.csv', index=False)

    torch.save(model.state_dict(), "./mobile_last.pt")

def acc_function(correct, total) :
    acc = correct / total * 100
    return acc

# def test() :
#     # 테스트 할때는 pretrain False
#     # model loader #  !!!!!! test할 때 !!!!!!! 추가
#     model.load_state_dict(torch.load("./mobile_best.pt", map_location=device))  # map_location은 cpu
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for i, (image, label) in enumerate(test_loader) :
#             images, labels = image.to(device), label.to(device)
#             output = model(images)
#             _, argmax = torch.max(output, 1)
#             total += images.size(0)
#             correct += (labels == argmax).sum().item()
#         acc = acc_function(correct, total)
#         print("acc for {} image : {:.4f}%".format(total, acc))

if __name__ == "__main__":
    train(best_val_acc)
    # test()