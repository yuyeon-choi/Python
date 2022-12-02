# NUMPY
## NumPy란?   
+ Data를 다루거나 Machine Learning을 활용하기 위해선 잘 만들어진 데이터의 구조가 필요   
+ NumPy는 가볍고 강력한 성능과 적절한 기능으로 Python으로 데이터를 다룰 때 필요한 배열을 제공함   
+ NumPy의 배열은 Python에서 제공하는 배열보다 빠르고 유연한 기능들을 제공   
+ NumPy는 차원의 개념을 잘 구현하고 있으며 1차원 데이터부터 n차원의 데이터까지 표현함   
+ Python을 사용하는 대부분의 환경에서 기본으로 설치되어 있음 그러나 설치가 되어있지 않다면, 다음의 명령어로 설치가능
```
pip install numpy
```
+ 데이터 사이언스 영역 대부분의 도구(Pandas) NumPy기반
***
### array 함수를 사용하여 배열 생성하기
```
import numpy as np

arr = np.array([1,2,3,4])
print(arr)
print(type(arr))
```
[1 2 3 4]   
<class 'numpy.ndarray'>

***

### 0으로 초기화된 배열

```
arr = np.zeros ((3,3))
print(arr)
```
[[0. 0. 0.]   
 [0. 0. 0.]   
 [0. 0. 0.]]

***

### 빈 값으로 만들어진 배열

```
arr = np.empty((4,4))
print(arr)
```
[[1.59938773e-316 3.80430547e-322 0.00000000e+000 0.00000000e+000]   
 [0.00000000e+000 1.50008929e+248 4.31174539e-096 9.80058441e+252]   
 [1.23971686e+224 2.59031995e-144 6.06003395e+233 1.06400250e+248]   
 [2.59050167e-144 5.22286946e-143 2.66023306e-312 0.00000000e+000]]   
***

### 1로 가득찬 배열을 만든다

```
arr = np.ones((3,3))

print(arr)
```
[[1. 1. 1.]   
 [1. 1. 1.]   
 [1. 1. 1.]]
***

### 배열의 생성

```
arr = np.arange(10)
print(arr)
```
[0 1 2 3 4 5 6 7 8 9]
***

### ndarray 배열의 모양, 차수, 데이터 타입 확인   
- shape: 배열의 모양
- ndim: 차원의 수
- dtype: 데이터 타입

##### 차원 확인하는 법 => [[ 수 = 차원

```
arr = np.array([[1,2,3],[4,5,6]])
print(arr)
```
[[1 2 3]   
 [4 5 6]]
```
arr.shape
arr.ndim
arr.dtype
```
(2, 3)   
2   
dtype('int64')
### 타입바꾸기
```
arr_float = arr.astype(np.float64)
arr_float.dtype
```
dtype('float64')
 

```
arr_str = np.array(['1','2','3'])
arr_str.dtype
```
dtype('<U1')   

```
arr_int = arr_str.astype(np.int64)
arr_int.dtype
```
dtype('int64')
***

### ndarry 배열의 연산
- 기본 연산자와 함수를 통해 배열 연산하기   
- 형태가 같은 배열 사이의 연산은 같은 위치의 요소들 사이 연산이 이루어짐

```
arr1 = np.array([[1,2],[3,4]])
arr2 = np.array([[5,6],[7,8]])
```

```
arr1 + arr2
```
array([[ 6,  8],
       [10, 12]])
```
np.add(arr1, arr2)
```
array([[ 6,  8],   
       [10, 12]])

```
arr1 * arr2
```
array([[ 5, 12],   
       [21, 32]])
```
np.multiply(arr1, arr2)
```
array([[ 5, 12],   
       [21, 32]])
***

#ndarray 배열 슬라이싱 하기

```
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
```

***

 123    [:2]  -> 123/ 456
 456    [1:3] -> 23/ 56/ 89
 789

```
arr_1 = arr[:2,1:3]
print(arr_1)
```

[[2 3]   
[5 6]]

***

```
arr[0,2]
```
3

***

 123    [0,1,2]  -> 모두 선택(큰배열 선택)   
 456    [2,0,1]  -> 첫번째 배열의 2번째요소/ 두번째 배열의    0번째 요소... (배열안의 요소 선택)   
 789

```
arr[[0,1,2],[2,0,1]]
```
array([3, 4, 8])
***

```
idx = arr > 3
print(idx)
```

> RESULT   
> [[False False False]   
>  [ True  True  True]   
>  [ True  True  True]]

***

```
print(arr[idx])
```
[4 5 6 7 8 9]
***
## Wine Quality 데이터
|No|변수명|변수 설명|
|------|---|---|
|1|fixed acidity|결합산도|
|2|volatile|휘발성산|
|3|citric acid|시트로산|
|4|residual sugar|발효 후 와인 속에 남아 있는 당분|
|5|chlorides|염화물|
|6|free sulfur dioxide|유리 이산화황|
|7|total sulfur dioxide|총 이산화황|
|8|density|농도|
|9|pH|산도|
|10|sulphates|황산염|
|11|alcohol|알코올|
|12|quality|품질|

```
redwine = np.loadtxt(fname = 'winequality-red.csv', delimiter=';', skiprows=1)
print(redwine)
```
[[ 7.4    0.7    0.    ...  0.56   9.4    5.   ]   
 [ 7.8    0.88   0.    ...  0.68   9.8    5.   ]   
 [ 7.8    0.76   0.04  ...  0.65   9.8    5.   ]   
 ...   
 [ 6.3    0.51   0.13  ...  0.75  11.     6.   ]   
 [ 5.9    0.645  0.12  ...  0.71  10.2    5.   ]   
 [ 6.     0.31   0.47  ...  0.66  11.     6.   ]]
***
## 기초 통계 분석
|No|매서드|매서드 설명|
|------|---|---|
|1|sum|배열 전체 혹은 특정 축에 대한 모든 원소의 합을 계산|
|2|mean|산술 평균을 계산|
|3|std|표준 편차를 계산|
|4|var|분산을 계산, std의 제곱과 같다|
|5|min|최소값|
|6|max|최대값|
### 합계

```
print(redwine.sum())
```
152084.78194
***

### 평균

```
print(redwine.mean())
```
7.926036165311652
***

### 축(axis)

```
print(redwine.sum(axis=0))
print(redwine.mean(axis=0))
```
[13303.1       843.985     433.29     4059.55      139.859   25384.   
 74302.       1593.79794  5294.47     1052.38    16666.35     9012.     ]
***

### [전체데이터:0번째 컬럼(세로줄)]

```
redwine[:,0].mean()
```
8.31963727329581
***

### 각 컬럼별 최대값

```
redwine.max(axis=0)
```
array([ 15.9    ,   1.58   ,   1.     ,  15.5    ,   0.611  ,  72.     ,   
       289.     ,   1.00369,   4.01   ,   2.     ,  14.9    ,   8.     ])

### 각 컬럼별 최소값

```
redwine.min(axis=0)
```
array([4.6    , 0.12   , 0.     , 0.9    , 0.012  , 1.     , 6.     ,   
       0.99007, 2.74   , 0.33   , 8.4    , 3.     ])

## 핵심정리_NumPy
1. NumPy는 다차원 배열을 쉽게 처리하고 효율적으로 사용할 수 있도록 지원하는 데이터 과학 도구의 핵심 패키지   
2. ndarray: NumPy에서 사용되는 기본 데이터 구조   
(N dimension array, n차원 행렬)   
3. NumPy의 기본적인 배열 통계 매서드를 통해 데이터의 기초적인 통계 분석을 할 수 있음
