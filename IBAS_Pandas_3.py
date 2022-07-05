import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn


dataframe = pd.read_csv("./DataSet/titanic.csv", encoding = "cp949", index_col = 0)
print(dataframe)
# 타이타닉 호에서의 생존자, 사망자 자료
# SibSp 는 2촌 관계(형제, 배우자)인 탑승자가 있는지, Parch 는 1촌 관계(부모, 자녀)인 탑승자가 있는지
# Pcalss 는 어떤 등급의 호실을 이용했는지, Cabin 은 어느 호실을 이용했는지, Embarked는 어느 지역에서 승선한 승객인지를 나타내는 변수
Condition = (dataframe["Age"] <= 14) & (dataframe["Parch"] == 0)
print(Condition)
# 특정 조건을 만족하는 객체만 가져올 수도 있음, 이때는 비교연산자(&&, == 등)이 아닌 조건연산자(&) 사용
np.median(dataframe[dataframe["Pclass"] == 1]["Fare"])
# numpy를 이용해 특정 조건에 해당하는 객체들의 기술통계량을 확인 가능


print(dataframe.isnull().sum())
# 나이, 객실, 승선지 변수에서 결측치 확인
# 결측치 처리에는 아래와 같은 방법이 있음
# 1. 평균값 또는 중앙값 또는 최빈값으로 대체
# 2. 행 제거 -> 해당 객체를 분석에서 제외
# 3. 열 제거 -> 해당 변수를 분석에서 제외
# 4. 대체값으로 대체 -> 결측치임을 나타내는 값으로 대체
print(dataframe["Embarked"].value_counts(normalize = True))
print(dataframe[dataframe["Embarked"].isna()])
dataframe.loc[62, "Embarked"] = 'S'
dataframe.loc[830, "Embarked"] = 'S'
# 승선지 변수는 중요도가 떨어지고 결측치 개수가 적으므로 최빈값인 S로 대체
dataframe["Cabin"].fillna('-', inplace = True)
# 객실 변수는 중요도가 높고 결측지 개수가 매우 많으므로 값이 없음을 나타내는 '-'문자로 대체
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1)
print(dataframe["Age"])
dataframe["Age"].fillna(0, inplace = True)
dataframe["Age"] = np.ceil(dataframe["Age"]) # np.round -> 반올림, np.ceil -> 올림, np.trunc -> 버림
# 나이 변수는 값이 없음을 나태는 0으로 대체하고 소숫점이 존재하는 경우 올림처리

dataframe_forVisualize = dataframe.copy() # copy를 쓰면 원래 데이터프레임에 영향을 주지 않음
dataframe_forVisualize.drop(['Name', 'Ticket'], axis = 1, inplace = True)
dataframe_forVisualize['Survived'].replace({0 : 'Dead', 1 : 'Survived'}, inplace = True)
dataframe_forVisualize['Embarked'].replace({'S' : 'SouthHampton', 'C' : 'CherBourg', 'Q' : 'QueensTown'}, inplace = True)
dataframe_forVisualize['Pclass'].replace({1 : 'First', 2 : 'Second', 3 : 'Third'}, inplace = True)
dataframe_forVisualize.rename(columns = {'Survived' : 'Status', 'Pclass' : 'Cabin Class', 'SibSp' : 'Companion(Sibling and Spouse)',
                                         'Parch' : 'Companion(Parents and Children)'}, inplace = True)
# 시각화를 용이하게 하기 위해 변수 이름 변경

dataframe_forVisualize.loc[dataframe_forVisualize["Age"] == 0, 'Age Group'] = '-'
dataframe_forVisualize.loc[(dataframe_forVisualize["Age"] > 0) & (dataframe_forVisualize["Age"] < 12), 'Age Group'] = 'under 12'
dataframe_forVisualize.loc[(dataframe_forVisualize["Age"] > 12) & (dataframe_forVisualize["Age"] < 20), 'Age Group'] = '12 - 20'
dataframe_forVisualize.loc[(dataframe_forVisualize["Age"] > 20) & (dataframe_forVisualize["Age"] < 40), 'Age Group'] = '20 - 40'
dataframe_forVisualize.loc[(dataframe_forVisualize["Age"] > 40) & (dataframe_forVisualize["Age"] < 60), 'Age Group'] = '40 - 60'
dataframe_forVisualize.loc[dataframe_forVisualize["Age"] > 60, 'Age Group'] = 'above 60'
print(dataframe_forVisualize["Age Group"].value_counts())
# 이산형 변수인 나이를 범주형 변수인 연령대 변수로 바꿈


sbn.countplot(x = dataframe_forVisualize["Status"]) # 승객의 생존 여부를 나타내는 막대 그래프
sbn.countplot(x = dataframe_forVisualize["Cabin Class"], order = ['First', 'Second', 'Third']) # 승객의 객실 등급을 나타내는 막대 그래프
# 지정한 순서대로 막대가 표시되게 끔할 수 있음
sbn.countplot(x = dataframe_forVisualize["Age Group"], order = ['under 12', '12 - 20', '20 - 40', '40 - 60', 'above 60']) # 승객의 연령대를 나타내는 막대 그래프
sbn.countplot(x = dataframe_forVisualize["Companion(Sibling and Spouse)"]) # 승객의 2촌 관계 동반자수를 나타내는 막대 그래프
sbn.countplot(x = dataframe_forVisualize["Companion(Parents and Children)"]) # 승객의 1촌 관계 동반자수를 나타내는 막대 그래프
sbn.countplot(x = dataframe_forVisualize["Embarked"]) # 승객의 출항지를 나타내는 막대그래프
sbn.countplot(x = dataframe_forVisualize["Sex"]) # 승객의 성별을 나타내는 막대그래프

sbn.countplot(x = dataframe_forVisualize["Cabin Class"], hue = dataframe_forVisualize["Status"]
              , order = ['First', 'Second', 'Third'], hue_order = ['Survived', 'Dead']) # 승객의 객실 등급을 범주로 하여 승객의 생존 여부를 표시
sbn.countplot(x = dataframe_forVisualize["Age Group"], hue = dataframe_forVisualize["Status"]
              , order = ['under 12', '12 - 20', '20 - 40', '40 - 60', 'above 60'], hue_order = ['Survived', 'Dead']) # 승객의 연령대를 범주로 하여 승객의 생존 여부를 표시
sbn.countplot(x = dataframe_forVisualize["Companion(Sibling and Spouse)"]
              , hue = dataframe_forVisualize["Status"], hue_order = ['Survived', 'Dead']) # 승객의 2촌 관계 동반자수를 범주로 하여 승객의 생존 여부를 표시
sbn.countplot(x = dataframe_forVisualize["Companion(Parents and Children)"]
              , hue = dataframe_forVisualize["Status"], hue_order = ['Survived', 'Dead']) # 승객의 1촌 관계 동반자수를 범주로 하여 승객의 생존 여부를 표시
sbn.countplot(x = dataframe_forVisualize["Embarked"]
              , hue = dataframe_forVisualize["Status"], hue_order = ['Survived', 'Dead']) # 승객의 출항지를 범주로 하여 승객의 생존 여부를 표시
sbn.countplot(x = dataframe_forVisualize["Sex"]
              , hue = dataframe_forVisualize["Status"], hue_order = ['Survived', 'Dead']) # 승객의 성별을 범주로 하여 승객의 생존 여부를 표시

print(dataframe_forVisualize[dataframe_forVisualize["Embarked"] == 'SouthHampton']["Fare"].median())
print(dataframe_forVisualize[dataframe_forVisualize["Embarked"] == 'CherBourg']["Fare"].median())
print(dataframe_forVisualize[dataframe_forVisualize["Embarked"] == 'QueensTown']["Fare"].median())
# 출항지별 요금 중앙값은 CherBourg > SouthHampton > QueensTown이므로 금액으로 인해 출항지별 사망자 비율이 차이나는 것이 아님
Graph = sbn.boxplot(x = dataframe_forVisualize["Embarked"], y = dataframe_forVisualize["Fare"]) # 승객의 출항지를 범주로 하여 객실 요금의 상자 그림을 표시
Graph.set(ylim = (0, 200)) # 결측치를 0으로 대체한 것을 고려해 Y축의 범위 조절

print(dataframe_forVisualize[dataframe_forVisualize['Status'] == "Surviced"]['Cabin Class'].value_counts()) # 객실 등급별 생존자수 확인
print(dataframe_forVisualize['Cabin Class'].value_counts()) # 객실 등급별 탑승자수 확인
print(dataframe_forVisualize[dataframe_forVisualize['Status'] == "Surviced"]['Age Group'].value_counts()) # 승객 연령대별 생존자수 확인
print(dataframe_forVisualize['Age Group'].value_counts()) # 승객 연령대별 탑승자수 확인
print(dataframe_forVisualize[dataframe_forVisualize['Status'] == "Surviced"]['Sex'].value_counts()) # 승객 성별에 따른 생존자수 확인
print(dataframe_forVisualize['Sex'].value_counts()) # 승객 성별에 따른 탑승자수 확인
# 이를 통해 연령대별, 객실 등급별, 성별별 생존율(생존자 / 탑승자)를 계산할 수 있음
# 남녀 성별에 따른 생존율 차이는 약 0.5
# 1등석 승객과 비 1등석 승객의 생존율 차이는 약 0.3
# 어린이 승객과 비 어린이 승객의 생존율 차이는 약 0.2