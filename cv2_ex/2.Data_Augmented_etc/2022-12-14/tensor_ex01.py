'''
    데이터 증강 라벨링 이미지 처리 데이터셋 구현 
    94p
''' 

import torch
import numpy as np

# # 1
# data = [[1, 2], [3, 4]]

# x_data = torch.tensor(data)
# print(x_data)
# print(x_data.shape)

# -----------------------------------------------------------------
# # 2 
# # numpy 배열로부터 생성 => numpy 버전에 따라 달라서 확인해 봐야함

# np_array = np.array(data).reshape
# x_np = torch.from_numpy(np_array)
# print(x_np)

# -----------------------------------------------------------------
# # 3
# data = [[1, 2], [3, 4]]

# x_data = torch.tensor(data)

# x_ones = torch.ones_like(x_data)
# print(f"Ones Tensor >> \n", x_ones)

# x_rand = torch.rand_like(x_data, dtype=torch.float)
# print(f"random Tensor >> \n", x_rand)


''' result
Ones Tensor >> 
 tensor([[1, 1],
        [1, 1]])
random Tensor >>
 tensor([[0.1900, 0.0922],
        [0.2202, 0.4934]])
'''

# -----------------------------------------------------------------
# # 4
# shape = (3, 3)      # 맘대로 지정
# randn_tensor = torch.rand(shape)
# ones_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)

# print(f"Random Tensor >> \n {randn_tensor} \n")
# print(f"Ones Tensor >> \n {ones_tensor} \n")
# print(f"Zeros Tensor >> \n {zeros_tensor} \n")

'''
Random Tensor >> 
 tensor([[0.1002, 0.6093, 0.0691],
        [0.1678, 0.1177, 0.2837],
        [0.6602, 0.1228, 0.9137]])

Ones Tensor >> 
 tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor >>
 tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
'''

# -----------------------------------------------------------------
# # 5
# tensor = torch.rand(3, 4)
# print(f"shape of tensor: {tensor.shape}")
# print(f"data type of tensor: {tensor.dtype}")
# print(f"device tensor is stored on: {tensor.device}")

'''
shape of tensor: torch.Size([3, 4])
data type of tensor: torch.float32   => 딥러닝 시 속도를 높이고 싶으면 낮추고(float32 -> float18) 성능을 높이고 싶으면 높인다(float32 -> float64). 
device tensor is stored on: cpu
'''

# -----------------------------------------------------------------
# # 6
# tensor = torch.rand(3, 4)
# if torch.cuda.is_available():
#     tensor = tensor.to("cuda")
# else:
#     tensor = tensor.to("cpu")
    
# print(f"Device tensor is stored on: {tensor.device}")

'''
Device tensor is stored on: cuda:0
''' 

# -----------------------------------------------------------------
# 7
tensor = torch.ones(4, 4)
tensor[:, 1] = 3
print(tensor)