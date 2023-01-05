import numpy as np

Arr1 = np.random.rand(1000000)
Arr2 = np.random.rand(1000000)
ArrVec = np.dot(Arr1, Arr2)
# 두 배열을 벡터화, 이중 for 문보다 훨씬 효율적

ArrExp = np.exp(Arr1)
ArrLog = np.log(Arr1)
ArrAbs = np.abs(Arr1)
ArrMax = np.max(Arr2, 0)
# 내장 함수를 통해 for 문 없는 효율적 계산 가능

ArrPlus = Arr2 + 1
# BroadCasting 은 연산자 통해 벡터/행렬에 대한 효율적 연산 가능
# 자세한 내용은 Numpy 문서를 참조

ArrRecommend = np.random.rand(5, 1)
# 벡터/행렬을 선언할 때는 행과 열의 크기를 모두 설정, 랭크가 1이 되지 않게해 오작동 최소화
# shape 함수는 벡터/행렬의 크기를 출력
# assert, reshape 함수는 벡터/행렬의 크기를 다시 정의
# 위의 함수는 시간 및 리소스 소모가 적기 때문에 필요할 때 마다 써주는 게 좋음
