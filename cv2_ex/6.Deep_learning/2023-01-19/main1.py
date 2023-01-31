import copy
import os.path
import time

import torch
import torchvision
from customdata import my_customdata
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = A.Compose([
    A.SmallestMaxSize(max_size=224),
    A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.08, rotate_limit=20, p=0.8),
    A.RandomShadow(p=.6),
    A.HorizontalFlip(p=.5),
    A.VerticalFlip(p=.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224,0.255)),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.SmallestMaxSize(max_size=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.255)),
    ToTensorV2()
])

## dataset과 dataloader는 한세트다.
# dataset
train_dataset = my_customdata("./dataset/train/", transform=train_transforms)
val_dataset = my_customdata("./dataset/valid/", transform=val_transforms)

# dataloader(항상 dataset이 필요함. 그래서 둘은 세트)
train_loader = DataLoader(train_dataset, batch_size=360, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=360, shuffle=False,
                        num_workers=2, pin_memory=True)

### facebookresearch/deit:main','deit_tiny_patch16_224 모델###########
model = torch.hub.load('facebookresearch/deit:main',
                       'deit_tiny_patch16_224', pretrained=True) # 마지막 test때는 False로 바꿈
model.head = nn.Linear(in_features=192, out_features=100)
model.to(device)
######################################################################

####### resnet50 ##################################################
# model = torchvision.models.resnet50(pretrained=True) # 마지막 test때는 False로 바꿈
# model.fc = torch.nn.Linear(in_features=2048, out_features=100)
# model.to(device)
######################################################################

####### resnet50 ##################################################
# model = torchvision.models.googlenet(pretrained=True) # 마지막 test때는 False로 바꿈
# model.fc = torch.nn.Linear(in_features=1024, out_features=100)
# model.to(device)
###############################################################

# #######shufflenet_v2_x0_5  ##################################################
# model = torchvision.models.shufflenet_v2_x0_5(pretrained=True) # 마지막 test때는 False로 바꿈
# model.fc = torch.nn.Linear(in_features=1024, out_features=100)
# model.to(device)
# ##############################################################################
# facebookresearch/deit:main','deit_tiny_patch16_224 모델 optimizer
# optimizer = torch.optim.Adam(model.head.parameters(), lr=0.01) # 이 경우 특이하게 model.head단에서 parameter를 땡겨오게 된다.(종종 있음)
# 다른 것들과 달리 구조가 다르기 때문. model.head 밑에는 RNN으로 되어 있어서 .head를 빼면 아까처럼 acc가 너무 낮은 것.
# AdamW와는 상성이 잘 맞지 않았음.


criterion = LabelSmoothingCrossEntropy()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# lr scheduler
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3,gamma=0.95) # gamma : 얼마나 떨어뜨릴지.

def train(model, criterion, train_loader, val_loader, optimizer, scheduler, num_epochs=10, device=device) :
    total = 0
    best_loss = 9999
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs) :
        print(f"Epoch {epoch} / {num_epochs - 1}")
        print("-"*10)

        for index, (image, label) in enumerate(train_loader) :
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = criterion(output, label)
            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(output, 1)
            acc = (label == argmax).float().mean()
            total += label.size(0)

            if (index + 1) % 10 == 0 :
                print("Epoch [{}/{}], Step [{}/{}], Loss {:.4f}, Acc {:.2f}".format(
                    epoch + 1, num_epochs, index+1, len(train_loader), loss.item(),
                    acc.item() * 100
                ))
        aveg_loss, val_acc = validation(epoch, model, val_loader, criterion, device)
        if aveg_loss < best_loss :
            best_loss = aveg_loss
            best_model_wts = copy.deepcopy(model.state_dict())  # best_model_wts 저장용 or loader용으로 쓰이지만 보통 loader용으로 쓰임
            save_model(model, save_dir="./")

    time_elapsed = time.time()  - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 6
    ))
    model.load_state_dict(best_model_wts)

def validation(epoch, model, val_loader, criterion, device) :
    print("Start validation # {}" .format(epoch+1))

    model.eval()
    with torch.no_grad() :
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        batch_loss = 0

        for i, (imgs, labels) in enumerate(val_loader) :
            imags, labels = imgs.to(device), labels.to(device)
            output = model(imags)
            loss = criterion(output, labels)
            batch_loss += loss.item()

            total += imags.size(0)
            _, argmax = torch.max(output, 1)
            correct += (labels == argmax).sum().item()
            total_loss += loss
            cnt +=1

    avrg_loss = total_loss / cnt
    val_acc = (correct / total * 100)
    print("Validation #{} Acc : {:.2f}% Average Loss : {:.4f}".format(epoch + 1,correct / total * 100, avrg_loss
    ))

    return avrg_loss, val_acc

def save_model(model, save_dir, file_name = "best.pt") :
    output_path = os.path.join(save_dir,file_name)
    torch.save(model.state_dict(), output_path)

if __name__ == "__main__" :
    train(model, criterion,train_loader, val_loader, optimizer,
          scheduler=exp_lr_scheduler, device=device)