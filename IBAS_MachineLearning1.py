# 1 . Machine Learning?
# Input → Logic → Output 에서 기존 Computer Sceince는 적절한 Output이 나오도록 Logic을 만드는 것
# Machine Leanring은 Input과 Output만 가지고도 Logic을 인간의 개입 없이 모델링하는 것

# 2. Machine Learning의 종류와 데이터의 중요성
# 지도학습 - 분류, 회귀, 추천 시스템 등 / Input과 Output 둘 다 존재
# 비지도 학습 - 군집화, 차원축소, 토픽 모델링 등 / Input만 존재
# 머신러닝은 데이터에 매우 의존적, Garbage In Garbage Out
# 학습 데이터에만 최적화되고 실제 데이터에는 과적합될 위험성
# 데이터만 넣는다고 최적의 결과값 도출 X, 데이터 특성 파악 및 최적 파라미터와 알고리즘을 구성하는 능력 필요
# 다양하고 광대한 데이터를 보유한 머신러닝은 더 좋은 품질 약속

import sklearn
import numpy as np
import pandas as pd
import xgboost
import lightgbm
from sympy.abc import lamda

# 3. Numpy의 ndarray 기본
# ndarray - n차원 배열 객체
# 1차원 - [ 1, 2, 3, 4 ] => 선 형태
# 2차원 - [ 1, 2, 3, 4 ]
#        [ 5, 6, 7, 8 ] => 직사각형 형태
# 3차원 - [ 1, 2, 3, 4 ]
#        [ 5, 6, 7, 8 ]이 여러겹 => 직육면체 형태
# 리스트 형태로 구현되지만 리스트보다 빅데이터 프로세싱에 훨씬 호율적
# 같은 데이터 타입만 들어갈 수 있음, astype()을 통해 형변환 가능, 형변환을 통해 메모리 절약가능
# ndarry는 행, 렬, 깊이가 아닌 axis 단위로 구분 = axis 0, axis 1, axis 2.....
# 1차원의 행은 axis 0, 2차원의 행렬은 axis 1 / axis 0, 3차원의 행,렬, 깊이는  axis 2 / axis 1 / axis 0
# 가장 기본이 되는 축이 axis 0이고 기본적인 축이 여러개 있는 축으로 갈수록 숫자가 높아진다고 생각하면 됨
array_1d = np.array([1,2,3])
array_2d = np.array([[4,5,6], [7,8,9]])
print(np.shape(array_1d))
print(np.shape(array_2d))
print(np.ndim(array_2d))
print(array_1d.astype(np.str_))

array_0to9 = np.arange(10)
print(array_0to9) # 0부터 9까지 자연수로 1차원 배열 생성
array_zero3by2 = np.zeros((3,2), dtype='int32')
print(array_zero3by2) # 0으로 3, 2 배열 생성
array_one3by2 = np.ones((3,2), dtype='int32')
print(array_one3by2) # 1로 3, 2 배열 생성
array_reshape2by5 = np.reshape(array_0to9, (2,5))
print(array_reshape2by5) # 2, 5 배열로 reshape
array_reshape_minus1by5 = np.reshape(array_0to9, (-1,5))
print(array_reshape_minus1by5) # 각 Column이 5개가 되도록 2차원 reshape
array_reshape_minus1by1 = np.reshape(array_0to9, (-1,1))
print(array_reshape_minus1by1) # 각 Column이 1개가 되도록 2차원 reshape
array_reshape_dimension1 = np.reshape(array_reshape_minus1by1, (-1, ))
print(array_reshape_dimension1) # 1차원으로 변환

print(array_reshape_dimension1[3])
print(array_reshape2by5[1,4])
print(array_reshape_dimension1[-2]) # 단일값 추출 인덱싱
print(array_reshape_dimension1[0:3])
print(array_reshape_dimension1[3:])
print(array_reshape_dimension1[:4])
print(array_reshape2by5[0:2, 2:4])
print(array_reshape2by5[1:, :3]) # 슬라이싱 인덱싱 - 연속된 인덱스 구간의 값을 추출
print(array_reshape_dimension1[[2,4,7]])
print(array_reshape2by5[[0,1]])
print(array_reshape2by5[[0,1], 2])
print(array_reshape2by5[[0,1], 3:]) # 팬시 인덱싱 - 연속되지 않아도 되는 복수의 인덱스의 값 추출
print(array_reshape2by5[array_reshape2by5 > 5])
print(array_reshape2by5[array_reshape2by5 % 2 == 0]) # 불린 인덱싱 - 특정 조건을 만족하는 값을 추출

array_sorted = np.sort([0, 3, 2, 9, 4, 5, 8, 7, 1, 6])[::-1]
print(array_sorted) # np.sort()는 특정 배열의 원본은 그대로 두고 그 배열을 정렬하여 리턴
array_sorted.sort()
print(array_sorted) # ndarray.sort()는 원본 배열을 정렬하고 값은 리턴하지 않음
array_sorted = [[8, 12], [7, 1]]
print(np.sort(array_sorted, axis=0)) # axis = 0 또는 Row 기준으로 정렬, 위에서 아래로 오름차순
print(np.sort(array_sorted, axis=1)) # axis = 1 또는 Column 기준으로 정렬, 좌에서 우로 오름차순
array_sorted = [3,1,9,5]
print(np.argsort(array_sorted, axis=0)) # 정렬한 후 정렬되기 전 원본 행렬의 인덱스를 리턴
# 정렬 전 인덱스 : 0 1 2 3 --정렬--> 정렬 후 인덱스 : 1 0 3 2
array_name = np.array(['John', 'Mike', 'Sarah', 'Kate', 'Samuel'])
array_score= np.array([78, 95, 84, 98, 88])
sort_indices_asc = np.argsort(array_score)
print(sort_indices_asc)
print(array_name[sort_indices_asc]) # argsort를 통해 mapping된 서로 다른 array도 동시에 정렬 가능

array_dot1 = [[1,2,3,], [4,5,6]]
array_dot2 = [[7,8], [9,10], [11,12]]
print(np.dot(array_dot1, array_dot2)) # 두 행렬(형태인 ndarray)의 내적
print(np.transpose(array_dot1))
print(np.transpose(array_dot2)) # 두 행렬의 전치행렬

# 4. Pandas의 dataframe 기본
# 시계열 데이터(금융 데이터 특징), 정형 데이터, 데이터 시각화에 유리, 1~2차원 행렬 데이터 처리 가능
titanic_df = pd.read_csv('./DataSet_MachineLearning/Ch1/titanic/train.csv')
# dataframe 기본 API는 Numpy의 ndarray와 동일, 추가로 display 옵션 지정 및 통계정보 출력 가능
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', 100)
print(titanic_df.describe())
# dataframe의 수정
titanic_df['New_col'] = 0
titanic_df['New_col_10'] = titanic_df['New_col']*10
titanic_df['New_col2'] = titanic_df['New_col']*2 + titanic_df['New_col_10']
titanic_df.drop('New_col', axis=1, inplace=True) # Inplace가 False면 원본 df는 두고 수정된 df 반환, True면 반환하지 않고 기존 df를 수정
# Dictionar, yndarray 및 리스트를 daraframe으로 상호전환 가능
dic_df = { 'Name' : ['A', 'B', 'C', 'D', 'E', 'F'], 'Value' : [1, 2, 3, 4, 5, 6], 'Gender' : ['Male', 'Male', 'Male', 'Female', 'Female', 'Female'] }
dictodf_df1 = pd.DataFrame(dic_df, columns=['Name', 'Value', 'Gender', 'Age'])
dictodf_df2 = pd.DataFrame(dic_df, index=['A', 'B', 'C', 'D', 'E', 'F'])
list_df = [1, 2, 3, 4, 5, 6]
listtodf_df = pd.DataFrame(list_df, index=['A', 'B', 'C', 'D', 'E', 'F'], columns=['Value'])
ndarray_df = np.array([1, 2, 3, 4, 5, 6])
ndarraytodf_df = pd.DataFrame(ndarray_df, index=list_df, columns=['Value'])
dftondarray = ndarraytodf_df.values
dftolist = ndarraytodf_df.values.tolist()
dftodict = ndarraytodf_df.to_dict()
# valuecounts() 변수의 결측치를 제외한 도수분포표 반환
titanic_vc1 = titanic_df['Pclass'].value_counts()
titanic_vc2 = titanic_df.value_counts()
# index는 dataframe의 레코드를 고유하게 식별하되(RDBMS의 PK), 연산의 대상이 아님, 자료형은 상관 무
titanic_index = titanic_df.index
titanic_df.reset_index(inplace=True) # reset_index()나 초기 할당 이외에 index는 수정 불가
print(titanic_df[['Name']])
print(titanic_df[titanic_df['Age'] > 20])
print(titanic_df[titanic_df['Age'] > 20][['Name', 'Age']])
print(titanic_df[(titanic_df['Age'] > 20) & (titanic_df['Pclass'] == 1)])
print(titanic_df[0:2]) # dataframe 직접 indexing은 비추천, loc/iloc 함수 활용
print(titanic_df.iloc[0, 2])
print(titanic_df.iloc[:, -1])
print(titanic_df.iloc[:, :-1])
# iloc는 위치 기반 indexing, [행번호, 열번호]의 정수를 받아 행렬의 평면좌표상에서 반환할 위치 결정
print(titanic_df.loc[0, 'Name'])
# loc는 명칭 기반 indexing, [인덱스명, 컬럼명]을 받아 반환할 위치를 결정
# dataframe 정렬, 병합, Aggregation
titanic_df_sorted = titanic_df.sort_values(by = ['Age', 'Pclass'], ascending = True)
print(titanic_df.count())
print(titanic_df['Age'].mean())
print(titanic_df['Age'].median())
titanic_df_groupby1 = titanic_df.groupby(by = ['Pclass'])
titanic_df_groupby2 = titanic_df.groupby(by = ['Pclass'])[['Age', 'Name', 'Pclass']]
# groupby의 결과물은 dataframe groupby
titanic_df_aggregated1 = titanic_df_groupby1.head(10)
titanic_df_aggregated2 = titanic_df_groupby2.count()
agg_format = { 'Age' : 'max', 'SibSp' : 'sum', 'Fare' : 'mean'}
titanic_df_aggregated3 = titanic_df_groupby1.agg(agg_format)
titanic_df_aggregated4 = titanic_df_groupby1.agg(agg_max=('Age', 'max'), agg_min=('Age', 'min'), fare_min=('Fare', 'min'))
# Aggrecation 함수를 적용하여 dataframe으로 변환가능
# dataframe 결측치 처리
print(titanic_df.isna().head(10))
print(titanic_df.isna().sum())
titanic_df['Cabin'] = titanic_df['Cabin'].fillna('UNKNOWN')
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())
# dataframe 값 대체
print(titanic_df['Pclass'].nunique())
titanic_df['Sex'] = titanic_df['Sex'].replace({'Female' : 'Male', 'F' : 'M'})
# 파이썬 lamda 함수 활용
lambda_square = lambda x : x**2
titanic_df['Name_Lens'] = titanic_df['Name'].apply(lambda x : len(x))
titanic_df['Age_Group'] = titanic_df['Age'].apply(lambda x : 'Senior' if x > 60 else ('Adult' if x > 20 else 'Child'))