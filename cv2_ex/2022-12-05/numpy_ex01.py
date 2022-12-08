import numpy as np

# 단일 객체 저장 및 불러오기
array = np.arange(0, 10)
print(array)

'''
결과 
[0 1 2 3 4 5 6 7 8 9]
'''

# .npy 파일에다가 저장하기
np.save("C:/Users/yuyeon/Documents/GitHub/Python/이미지다루기및데이터셋구축실습/2022-12-05/save.npy", array)

# 불러오기
result = np.load('C:/Users/yuyeon/Documents/GitHub/Python/이미지다루기및데이터셋구축실습/2022-12-05/save.npy')
print("result >> ", result)

'''
결과
result >>  [0 1 2 3 4 5 6 7 8 9]
'''