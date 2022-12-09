import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.io import read_image

training_data = datasets.FashionMNIST(
    root="data", 
    train=True, 
    download=True, 
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False, 
    download=True, 
    transform=ToTensor()
)

img_size = 28
num_images = 5
with open('./data/FashionMNIST/raw/t10k-images-idx3-ubyte', 'rb') as f:
    a = f.read(16)
    buf = f.read(img_size*img_size*num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(float)
    data = data.reshape(num_images, img_size, img_size, 1)
    
    image = np.asarray(data[1]).squeeze()
    plt.imshow(image, 'gray')
    # plt.show()

with open('./data/FashionMNIST/raw/train-labels-idx1-ubyte', 'rb') as f:
  buf = f.read(num_images)
  labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
#   print(labels[1])

  labels_map = {0: 'T-Shirt',
              1: 'Trouser',
              2: 'Pullover',
              3: 'Dress',
              4: 'Coat',
              5: 'Sandal',
              6: 'Shirt',
              7: 'Sneaker',
              8: 'Bag',
              9: 'Ankle Boot'}

columns = 3
rows = 3
fig = plt.figure(figsize=(8, 8))

for i in range(1, columns * rows+1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item() # 랜덤한 값을 가져오고 사이즈는 1
    img, label = training_data[sample_idx]  
    fig.add_subplot(rows, columns, i)         
    plt.title(labels_map[label])
    plt.axis('off')
    plt.imshow(img.squeeze(), cmap='gray')
# plt.show()
# 해당 라벨과 이미지를 확인할 수 있다.

### save annotation csv
# header
img_size = 28

imgf = open('./data/FashionMNIST/raw/train-images-idx3-ubyte', 'rb')
imgd = imgf.read(16)
lblf = open('./data/FashionMNIST/raw/train-labels-idx1-ubyte', 'rb')
lbuf = lblf.read(8)
df_dict = {
    'file_name' : [],
    'label' : []
}
idx = 0
os.makedirs('./data/FashionMNIST/imgs', exist_ok=True)
# imgs 폴더를 생성하고 이미 생성된 파일이 있으면 다시 돌릴때 dierectory를 생성하지 않는다.
while True:     
    imgd = imgf.read(img_size*img_size)
    if not imgd:
        break
    data = np.frombuffer(imgd, dtype=np.uint8).astype(float)
    data = data.reshape(1, img_size, img_size, 1)
    image = np.asarray(data).squeeze()
    lbld = lblf.read(1)
    labels = np.frombuffer(lbld, dtype=np.uint8).astype(np.int64)
    file_name = f'{idx}.png'
    cv2.imwrite(f'./data/FashionMNIST/imgs/{file_name}', image)
    idx += 1
    df_dict['label'].append(labels[0])
    df_dict['file_name'].append(file_name)


#print(df_dict)
import pandas as pd
df = pd.DataFrame(df_dict)
# print(df)
df.to_csv('./annotations.csv')

### _____________________________________________________________________
### save test_annotation csv
test_img_size = 28

test_imgf = open('./data/FashionMNIST/raw/t10k-images-idx3-ubyte', 'rb')
test_imgd = test_imgf.read(16)
test_lblf = open('./data/FashionMNIST/raw/t10k-labels-idx1-ubyte', 'rb')
test_lbuf = test_lblf.read(8)
test_df_dict = {
    'file_name' : [],
    'label' : []
}
idx = 0
os.makedirs('./data/FashionMNIST/imgs2', exist_ok=True)
while True:     
    imgd = test_imgf.read(test_img_size*test_img_size)
    if not imgd:
        break
    data = np.frombuffer(imgd, dtype=np.uint8).astype(float)
    data = data.reshape(1, test_img_size, test_img_size, 1)
    image = np.asarray(data).squeeze()
    lbld = test_lblf.read(1)
    labels = np.frombuffer(lbld, dtype=np.uint8).astype(np.int64)
    file_name = f'{idx}.png'
    cv2.imwrite(f'./data/FashionMNIST/imgs2/{file_name}', image)
    idx += 1
    test_df_dict['label'].append(labels[0])
    test_df_dict['file_name'].append(file_name)

#print(df_dict)
import pandas as pd
df = pd.DataFrame(test_df_dict)
# print(df)
df.to_csv('./test_annotations.csv')
#______________________________________________________________________


class CustomImageDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotation_file, names=['file_name', 'label'], skiprows=[0])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) 
        try:
            image = read_image(img_path)

        except:
            print(self.img_labels.iloc[idx, 0])
            exit()
        label = int(self.img_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

#Define Neural Networks Model.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512) # nn.Linear(input size, output size)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)  # 최종으로 10가지의 분류가 나옴 (feature 값)
        
    def forward(self, x):
        x = x.float()
        h1 = F.relu(self.fc1(x.view(-1, 784))) # activation function 으로 relu 사용.
        # x.view(-1, 784) :  차원을 ?? x  784로 변경하라는 뜻
        # 28 x 28 => ?? x 784  | 따라서 ?? = 1
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        h5 = F.relu(self.fc5(h4))
        h6 = self.fc6(h5)
        return F.log_softmax(h6, dim=1) # 각각의 확률이 나옴. 
        # softmax는 0-1로 되어있는데, 퍼센트가 높을수록 (1에 가까울수록) 분류에 속할 가능성이 높다.
        # https://magicode.tistory.com/33 참조
        #loss(실제 라벨값) 측정 

'''
(활성화함수에 대해 다시 공부하기!)
입력 신호의 총합을 출력 신호로 변환하는 함수를 일반적으로 활성화 함수
입력 신호의 총합이 활성화를 일으키는지를 정하는 역할
'''
#Prepare Data Loader for Training and Validation
transform = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))])

# print("init model done")

epochs = 10 # 학습을 몇번할것인가?
lr = 0.01   # Learning rate : 업데이트 시킬때 한번에 어느정도 할 것인가? 
            # 적당한 러닝레이트를 찾아야한다!
momentum = 0.5 # 옵티마이저에 최적화함수에 들어가는것인데 관성이다.
               # 더 진행을 해보고 더욱 수렴하는 방향이 있는지 확인하라
no_cuda = True # 11:30 분쯤? 오전 마지막 타임에 설명하심 다시 찾아보기
seed = 1      
log_interval = 200  

use_cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

# print("set vars and device done")

# train_loader = torch.utils.data.DataLoader(
#   datasets.MNIST('../data', train=True, download=True,
#                  transform=transform),
#     batch_size = batch_size, shuffle=True, **kwargs)

batch_size = 64
test_batch_size = 1000

dataset = CustomImageDataset(
  annotation_file='./annotations.csv',
  img_dir='./data/FashionMNIST/imgs'
  )
# ___________________________________________________________________
test_dataset = CustomImageDataset(
  annotation_file='./test_annotations.csv',
  img_dir='./data/FashionMNIST/imgs2'
  )
#____________________________________________________________________

train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
    batch_size=test_batch_size, shuffle=True, **kwargs)

# test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=False, download=True,
#                  transform=transform),
#     batch_size=test_batch_size, shuffle=True, **kwargs)

model = Net().to(device)
# .to() : 어떤한 변수나 가중치들, 텐서로 이루어질수있는 파이토치 하에 애들을 보낼수있다. 
# (cpu로 사용할건지 gpu로 사용할지)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) # nll loss는 분류문제에서 사용되며 softmax함수와 같이 사용된다.
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(log_interval, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)     
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format
          (test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, 11):
    train(log_interval, model, device, train_loader, optimizer, epoch)
    test(log_interval, model, device, test_loader)
torch.save(model, './model2.pt')
