# Numpy 가장 많이 사용되는 함수
# 1. 원소 정렬
import numpy as np
# def -> 오름차순 정렬 형태
array = np.array([15, 20, 5, 12, 7])
np.save("C:/Users/yuyeon/Documents/GitHub/Python/이미지다루기및데이터셋구축실습/2022-12-05/array.npy", array)
array_data = np.load("C:/Users/yuyeon/Documents/GitHub/Python/이미지다루기및데이터셋구축실습/2022-12-05/array.npy")
array_data.sort()   # 데이터 정렬
print("오름차순 >> ", array_data)

# 내림차순 정렬
print("내림차순 >> ", array_data[::-1])

'''
[결과]
오름차순 >>  [ 5  7 12 15 20]
내림차순 >>  [20 15 12  7  5]
'''