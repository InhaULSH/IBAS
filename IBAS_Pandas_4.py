import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn


dataframe = pd.read_csv("./DataSet/house_price.csv", encoding = "cp949", index_col = 0)
# 부동산의 조건에 따른 부동산 가격 데이터


print(dataframe.isnull().sum())
dataframe.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'MicsYal'], axis = 1, inplace = True)
# 결측치가 1000개가 넘어가는 변수들 존재, 해당 열을 삭제
# 독립 / 종속 -> 수치 / 수치인 경우 상관계수나 산점도로 시각화 => 자동으로 결측치를 제외함
# 독립 / 종속 -> 범주 / 수치인 경우 상자 그림으로 시각화 => 결측치를 수동으로 처리해야함 => 결측치를 None이나 0으로 표시해야함

dataframe.loc[dataframe[dataframe['TotalBsmtSF'] == 0].index, 'hasBsmt'] = 'False'
dataframe.loc[dataframe[dataframe['TotalBsmtSF'] != 0].index, 'hasBsmt'] = 'True'
# 지하실 넓이가 0인 객체와 0이 아닌 객체의 인덱스를 불러와 해당 인덱스의 객체에 hasBsmt 값을 부여
dataframe.loc[dataframe[dataframe['Fireplaces'] == 0].index, 'hasFireplaces'] = 'False'
dataframe.loc[dataframe[dataframe['Fireplaces'] != 0].index, 'hasFireplaces'] = 'True'
# 벽난로 크기가 0인 객체와 0이 아닌 객체의 인덱스를 불러와 해당 인덱스의 객체에 hasFireplaces 값을 부여
dataframe.loc[dataframe[dataframe['PoolArea'] == 0].index, 'hasPoolArea'] = 'False'
dataframe.loc[dataframe[dataframe['PoolArea'] != 0].index, 'hasPoolArea'] = 'True'
# 수영장 크기가 0인 객체와 0이 아닌 객체의 인덱스를 불러와 해당 인덱스의 객체에 hasPoolArea 값을 부여
dataframe.loc[dataframe[(dataframe['OpenPorchSF'] == 0) & (dataframe['EnclosedPorch'] == 0)
                        & (dataframe['3SsnPorch'] == 0) & (dataframe['ScreenPorch'] == 0)].index, 'hasPorch'] = 'False'
dataframe.loc[dataframe[(dataframe['OpenPorchSF'] != 0) | (dataframe['EnclosedPorch'] != 0)
                        | (dataframe['3SsnPorch'] != 0) | (dataframe['ScreenPorch'] != 0)].index, 'hasPorch'] = 'True'
# 어떤 형태로는 발코니형 현관의 크기가 0인 객체와 어느 하나의 발코니형 현광의 크기라도 0이 아닌 객체의 인덱스를 불러와 해당 인덱스의 객체에 hasPoolArea 값을 부여
# df.loc[df[조건].index, '새 변수'] = 값 을 통해 수치형 변수를 범주형 변수로 바꾸면서 결측치를 가진 개체에 대한 오류 가능성을 없앨 수 있음

print('상자 그림으로 시각화할 범주형 변수', dataframe.dtypes[dataframe.dtypes == 'object'].index)
print('산점도로 시각화할 수치형 변수', dataframe.dtypes[dataframe.dtypes != 'object'].index)
# 범주형 변수와 수치형 변수를 확인
dataframe_Category = dataframe[['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
                                'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                                'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                                'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                                'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
                                'SaleType', 'SaleCondition', 'hasBsmt', 'hasFireplaces', 'hasPool', 'hasPrch', 'SalePrice']].copy()
dataframe_Number = dataframe[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                              'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'ToalBsmtSF', '1stFirSF', '2ndFirSF', 'LowQualFinSF',
                              'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                              'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                              'EnclosedPorch', '3SsnPrch', 'ScreenPrch', 'PoolArea', 'MoSolid', 'YrSold', 'SalePrice']].copy()
# 범주형 변수와 수치형 변수로 데이터 프레임 분리
dataframe_Category['MasVnrType'] = dataframe_Category['MasVnrType'].fillna('-')
dataframe_Category['BsmtQual'] = dataframe_Category['BsmtQual'].fillna('-')
dataframe_Category['BsmtCond'] = dataframe_Category['BsmtCond'].fillna('-')
dataframe_Category['BsmtExposure'] = dataframe_Category['BsmtExposure'].fillna('-')
dataframe_Category['BsmtFinType1'] = dataframe_Category['BsmtFinType1'].fillna('-')
dataframe_Category['BsmtFinType2'] = dataframe_Category['BsmtFinType2'].fillna('-')
dataframe_Category['Electrical'] = dataframe_Category['Electrical'].fillna('-')
dataframe_Category['FireplaceQu'] = dataframe_Category['FireplaceQu'].fillna('-')
dataframe_Category['GarageType'] = dataframe_Category['GarageType'].fillna('-')
dataframe_Category['GarageFinish'] = dataframe_Category['GarageFinish'].fillna('-')
dataframe_Category['GarageQual'] = dataframe_Category['GarageQual'].fillna('-')
dataframe_Category['GarageCond'] = dataframe_Category['GarageCond'].fillna('-')
# 범주형 변수의 결측치 처리