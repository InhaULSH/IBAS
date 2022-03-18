import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#판다스 기본 문법
#데이터 프레임 - 2차원의 자료구조 => 행렬
df = pd.DataFrame({"a":[4,5,6], "b": [1,2,3], "c":[7,8,9]}, index = [1,2,3])
print(df)
df_a0 = df[["a"]]
print(df_a0)

#시리즈 - 1차원의 자료구조 => 벡터
df_a = df["a"]
print(df_a)

#섭셋 - 조건에 따라 값을 Boolean형태로 가져올 수 있음
print(df_a > 4)
df_ab = df[["a", "b"]] #여러 열을 호출할때는 []로 한번더 감싸줘야함
print(df_ab > 4)

#데이터 요약
df_counts = df["a"].value_counts() #한 열에서 어떤 값이 몇번 나왔는지 세줌
print(df_counts)
df_discribe = df.describe() #각 열에 대한 통계수치를 보여줌
print(df_discribe)
df["a"].sum() #데이터프레임 전체 또는 부분의 합을 구함
df["a"].median() #데이터프레임 전체 또는 부분의 중앙값을 구함
df["a"].quantile([0.25, 0.75]) #데이터프레임 전체 또는 부분의 사분위수를 구함
df["a"].min()
df["a"].max() #데이터프레임 전체 또는 부분의 최대/최소를 구함
df["a"].mean() #데이터프레임 전체 또는 부분의 평균을 구함
df["a"].var()
df["a"].std() #데이터프레임 전체 또는 부분의 분산/표준편차를 구함

# #데이터 가공
df.rename({"b" : "block"}) #한 열의 이름을 바꿈
df_ColumSort = df["a"].sort_values() #한 열의 값을 정렬해줌
df_AllSort = df.sort_values("a") #데이터 프레임 전체를 한 열을 기준으로 값들을 정렬해줌
df_ReverseSort = df.sort_values("a", ascending = False) #데이터 프레임 전체를 한 열을 기준으로 값들을 역순으로 정렬해줌
df_IndexSort = df.sort_index() #데이터 프레임 전체를 인덱스를 기준으로 값들을 정렬해줌
print(df_ColumSort, df_AllSort, df_ReverseSort)
df = df.drop(["c"], axis = 1) #일정한 기준으로 행이나 열을 삭제함, 항상 삭제한 것을 원래 데이터 프레임에 대입해 줘야함
print(df)
pd.melt(df) #행에 있는 데이터를 열로 분배해줌
            #    행1 행2 행3        열1 행1 X
            # 열1 X   Y   Z    ==> 열1 행2 Y
            # 열2 A   B   C        열1 행3 Z
            #                      열2 행1 A ....

#데이터 집계
df_mean = df.groupby(["a"])["b"].mean() # a열을 기준으로하여 b열의 최솟값을 계산함
df_mean = df.groupby(by = ["a"]).sum() # "by = []" 를 이용해서 하나 이상의 열을 기준으로 나머지 열에 대한 값을 계산할 수 있음

#데이터 시각화
df_plot = df.plot()
df_plotBar = df.plot.bar() #데이터프레임을 다양한 형태의 그래프로 시각화 할 수 있음
print(df_plot, df_plotBar)

#데이터 파일 읽어들이기
df_new = pd.read_csv("data/국가_대륙_별_상품군별_온라인쇼핑_해외직접판매액_20210716133340.csv", encoding = "euc-kr") #같은 디렉토리 내의 파일 읽어들이기



#실제 데이터 다뤄보기
plt.rc('axes', unicode_minus = False) #글꼴 오류를 방지하기 위함

#파일을 불러오고 데이터프레임 정보 확인하기
df = pd.read_csv("data/소상공인시장진흥공단_상가업소정보_의료기관_201909.csv", low_memory = False)
print(df.shape) # 데이터프레임의 행렬 개수를 볼 수 있음
print(df.head()) # 데이터프레임의 앞 일부분을 볼 수 있음
print(df.tail()) # 데이터프레임의 뒤 일부분을 볼 수 있음
print(df.info()) # 파일의 정보를 볼 수 있음
print(df.columns)
print(df.dtypes) #컬럼의 이름과 데이터 타입을 볼 수 있음

#결측치를 확인하기
nullCount = df.isnull().sum()
print(nullCount) #결측치( = 빈 값을 가진 누락된 데이터)를 세어줌
nullCount.plot() #결측치를 선형그래프로 보여줌
nullCount.plot.bar() #결측치를 막대그래프로 보여줌
nullCount.plot.bar(rot = 60) #범례의 글자를 기울여 표시할 수 있음
nullCount.plot.barh(figsize = (5,7)) #세로 막대그래프를 보여주면서 그래프의 사이즈를 지정할 수 있음

#결측치가 많은 컬럼 삭제하기
df_null_count = nullCount.reset_index() #리셋인덱스를 통해 데이터프레임 형태로 바꿀 수 있음
print(df_null_count.head())
df_null_count.columns = ["컬럼명", "결측치수"] #데이터프레임의 컬럼명을 변경할 수 있음
print(df_null_count.head())
df_null_count_top = df_null_count.sort_values(by = ["결측치수"], ascending = False).head(10)
print(df_null_count_top)
drop_columns = df_null_count_top["컬럼명"].tolist() #한 열의 값들을 리스트로 만듬
df.drop(drop_columns, axis = 1) #조건에 따라 삭제할 수 있음, axis는 0일때 열을 기준으로 삭제, 1일때 행을 기준으로 삭제

#한 컬럼(열)의 값들을 확인하기
df["위도"].mean()
df["위도"].median()
df["위도"].max()
df["위도"].min()
df["위도"].describe()
df["상권업종대분류명"].unique() #값의 종류가 어떻게 되는지
df["상권업종대분류명"].nunique() #값의 종류의 개수를 무엇인지 출력해줌
df["상권업종중분류명"].unique()
df["상권업종중분류명"].nunique()
df["상권업종소분류명"].unique()
df["상권업종소분류명"].nunique() #이는 len(df["상권업종소분류명"].unique()) 와 같은 값임
df["시도명"].value_counts() #해당 열의 값이 몇개인지 세어줌
df["시도명"].value_counts(normalize = True) #해당 열에서 각 값이 차지하는 비율을 세어줌
df["시도명"].value_counts(normalize = True).plot.barh()

#한 컬럼(열)에 대한 그래프 그리기
sns.countplot(data = df, x = "시도명") #countplot은 X와 Y 중 하나만 넣어도 됨
varSanggwon = df["상권업종중분류명"].value_counts()
propSanggwon = df["상권업종중분류명"].value_counts(normalize=True)
varSanggwon.plot.bar(figsize = (7,8), grid = True)
propSanggwon.plot.pie(figsize = (7,7))

#파일 전체에서 데이터 색인하기
df["상권업종중분류명"] == "약국/한약방" #상권업종중분류명이 약국/한약방인지를 확인
(df["상권업종소분류명"] == "약국/한약방") & (df["시도명"] == "서울특별시") #논리 연산을 이용해 조건을 확인할 수도 있음
print(df[df["상권업종중분류명"] == "약국/한약방"]) # df["상권업종중분류명"] == "약국/한약방"을 만족하는 데이터만 가져옴
print(df[df["상권업종중분류명"] ==  "약국/한약방"].shape) # 위 데이터의 행렬 수 확인
df_medical = df[df["상권업종중분류명"] == "약국/한약방"].copy() #다른 목적으로 활용할 생각이라면 copy()를 써줘야함
print(df.loc[df["상권업종대분류명"] == "의료", "상권업종중분류명"]) # df["상권업종대분류명"] == "의료"를 만족하는 데이터의 상권업종중분류명을 가져옴
df.loc[df["상권업종대분류명"] == "의료", "상권업종중분류명"].value_counts()
M = df["상권업종대분류명"] == "의료"
print(df.loc[M, "상권업종중분류명"].value_counts()) #위의 상권업종중분류명에 해당하는 데이터 개수 새기

#데이터 전처리하기
df_seoul_drug = df[(df["상권업종소분류명"] == "약국") & (df["시도명"] == "서울특별시")].copy()
sdCounter = df_seoul_drug["시군구명"].value_counts()
sdCounter.head()
sdCounter.plot.bar(rot = 60)
sdProportion = df_seoul_drug["시군구명"].value_counts(normalize=True)
sdProportion.head()
sdProportion.plot.pie()
df_seoul_drug["상호명"].str.contains("약국") #상호명에 "약국"이 들어가 있는 데이터만 찾음
df_seoul_drug.loc[df_seoul_drug["상호명"].str.contains("약국"),"상호명"] #상호명에 "약국"이 들어가 있는 데이터를 가져옴
print(df_seoul_drug.shape)
drop_row1 = df_seoul_drug.loc[df_seoul_drug["상호명"].str.contains("약방"),"상호명"].index
drop_row1 = drop_row1.tolist()
drop_row2 = df_seoul_drug.loc[df_seoul_drug["상호명"].str.contains("스토어"),"상호명"].index
drop_row2 = drop_row2.tolist()
drop_row = drop_row1 + drop_row2
df_seoul_drug = df_seoul_drug.drop(drop_row, axis = 0)
print(df_seoul_drug)
#"약국"이 아닌 데이터를 일부 제거하는 과정 거침

#그래프로 표현하기
df_seoul = df[df["시도명"] == "서울특별시"]
print(df_seoul.shape)
print(df_seoul ["시도명"].value_counts())
df_seoul["시도명"].value_counts().plot.bar(figsize=(10, 4), rot=30)
plt.figure(figsize=(9, 8)) #seaborn을 사용하기 전에 미리 크기를 지정
sns.scatterplot(data=df_seoul, x="경도", y="위도", hue="상권업종중분류명") # X축은 경도, Y축은 위도, 범례는 상권업종중분류명인 점도표 생성
sns.scatterplot(data=df_seoul, x="경도", y="위도", hue="시군구명") # X축은 경도, Y축은 위도, 범례는 시군구명인 점도표 생성



# 실제 데이터 가설검정 해보기
import os
if os.name == "nt":
    sns.set(font="Malgun Gothic") #글꼴 설정
plt.rc("axes", unicode_minus=False) # 글꼴 설정

df2 = pd.read_csv("data/국민건강보험공단_건강검진정보_20191231.csv", encoding = "euc-kr")
print(df2.shape)
print(df2.head())
print(df2.sample(10))
print(df2.columns)
df2.groupby(['성별코드']).mean() #성별코드에 따라 각 열의 평균을 구함
df2.groupby(["성별코드"]).count() #성별코드에 따라 각 열의 개수를 카운트함
df2.groupby(["성별코드"])["가입자 일련번호"].count() #성별코드에 따라 가입자 일련번호의 개수를 카운트함
df2.groupby(["성별코드", "음주여부"])["가입자 일련번호"].count() #성별코드와 음주여부에 따라 가입자 일련번호의 개수를 카운트함
print(df2.groupby(["성별코드", "음주여부"])["감마 지티피"].describe())
print(df2.groupby(["성별코드", "음주여부"])["감마 지티피"].agg(["count", "mean", "median"])) # 두 명령어를 살펴보면 max값 때문에 평균이 편향되었음을 알 수 있음

df2.pivot #연산하지 않고 데이터의 구조를 바꿈
df2.pivot_table(index = "성별코드", values = "가입자 일련번호", aggfunc = "count")
df2.pivot_table(index = "음주여부", values = "가입자 일련번호", aggfunc = "count")
pd.pivot_table(df2, index = "음주여부", values = "감마 지티피", aggfunc = ["mean", "median"]) #groupby와 기능 유사함
pd.pivot_table(df2, index = "음주여부", values = "감마 지티피") #기본적으로는 평균을 구함

#데이터 시각화로 가설검정하기
his_df2 = df2.hist(figsize = (12,12)) # 히스토그램 그리기
his_df2 = df2.iloc[:,:12].hist(figsize = (12,12))
his_df2 = df2.iloc[:,12:24].hist(figsize = (12,12)) #특정 행과 열까지를 사용해서 히스토그램 그리기
his_df2 = df2.iloc[:,12:24].hist(figsize = (12, 12), bins = 100) #히스토그램의 막대 개수를 조절가능

df2_sample = df2.sample(1000, random_state = 1) # 1000개의 샘플을 무작위로 뽑아옴
sns.countplot(x = "음주여부", data = df2)
sns.countplot(x = "음주여부", data = df2, hue = "성별코드")
sns.set(font_scale = 1.5, font = "Malgun Gothic") # seaborn에서 그릴 그래프의 옵션을 조정할 수 있음
sns.countplot(data = df2, x = "연령대 코드(5세단위)", hue = "음주여부") #옵션 조정 후 그리는 그래프에는 바뀐 옵션이 그대로 적용됨
plt.figure(figsize = (15,4))
sns.countplot(data = df2, x = "신장(5Cm단위)", hue = "성별코드")
sns.countplot(data = df2, x = "체중(5Kg 단위)", hue = "성별코드") #성별에 따라 체중의 분포가 다름을 확인할 수 있음
sns.countplot(data = df2, x = "체중(5Kg 단위)", hue = "음주여부") #음주 여부에 따라 체중의 분포가 다름을 확인할 수 있음
sns.barplot(data = df2_sample, x = "연령대 코드(5세단위)", y = "총 콜레스테롤", hue = "음주여부") #빠른 연산을 위해 샘플로 시각화를 진행할 수 있음

plt.figure(figsize = (15,4))
sns.barplot(data = df2_sample, x = "연령대 코드(5세단위)", y = "트리글리세라이드", hue = "음주여부", ci = 95) # ci를 통해 신뢰구간을 지정할 수 있음, 음주여부에 따른 트리글리세라이드 수치의 분포가 다름을 알 수 있음
sns.barplot(data = df2_sample, x = "연령대 코드(5세단위)", y = "체중(5Kg 단위)", hue = "성별코드", ci = None) # 나이와 체중은 상관관계가 있음을 알 수 있음
sns.barplot(data = df2_sample, x = "연령대 코드(5세단위)", y = "체중(5Kg 단위)", hue = "음주여부", ci = None) # 음주와 체중은 상관관계가 있음을 알 수 있음

plt.figure(figsize = (15,4))
sns.lineplot(data = df2_sample, x = "연령대 코드(5세단위)", y = "체중(5Kg 단위)", hue = "성별코드") # 나이와 체중은 상관관계가 있음을 알 수 있음
sns.lineplot(data = df2, x = "연령대 코드(5세단위)", y = "신장(5Cm단위)", hue = "성별코드", ci = "sd")
sns.lineplot(data = df2, x = "연령대 코드(5세단위)", y = "신장(5Cm단위)", hue = "음주여부", ci = "sd") # lineplot에서는 신뢰구간이 그림자로 표시됨
sns.pointplot(data = df2, x = "연령대 코드(5세단위)", y = "신장(5Cm단위)", hue = "음주여부", ci = "sd") # pointplot에서는 막대를 통해 신뢰구간과 편차가 표시됨, 두 개의 그래프를 같이 그릴수도 있음
# #자료의 종류에 따라 적절한 그래프가 다름

sns.boxplot(data = df2, x = "신장(5Cm단위)", y = "체중(5Kg 단위)")
sns.boxplot(data = df2, x = "신장(5Cm단위)", y = "체중(5Kg 단위)", hue = "성별코드")
sns.boxplot(data = df2, x = "신장(5Cm단위)", y = "체중(5Kg 단위)", hue = "음주여부") # boxplot에서는 사분위수와 최대/최소 값 처럼 다양한 통계수치들을 박스로 보여줌
sns.violinplot(data = df2, x = "신장(5Cm단위)", y = "체중(5Kg 단위)")
sns.violinplot(data = df2, x = "신장(5Cm단위)", y = "체중(5Kg 단위)", hue = "성별코드") # violinplot에서는 boxplot에서보다 자세하게 수치의 분포를 보여줌
# sns.violinplot(data = df2_sample, x = "신장(5Cm단위)", y = "체중(5Kg 단위)", hue = "음주여부", split = True) # split 옵션을 통해 여러 범례의 분포를 붙여서 볼 수 있음
# sns.violinplot(data = df2_sample, x = "연령대 코드(5세단위)", y = "혈색소", hue = "음주여부", split = True)
sns.swarmplot(data = df2_sample, x = "신장(5Cm단위)", y = "체중(5Kg 단위)", hue = "음주여부") # swarmplot에서는 산점도를 통해 자료의 분포를 보여줌, 두 개의 그래프를 같이 그릴수도 있음
sns.lmplot(data = df2_sample, x = "연령대 코드(5세단위)", y = "혈색소", hue = "음주여부") # lmplot에서는 scatterplot에 더해 회귀선을 표시해줌
sns.lmplot(data = df2_sample, x = "연령대 코드(5세단위)", y = "혈색소", hue = "음주여부", col = "성별코드") # col 옵션을 통해 어떤 범례에 따라 색상을 다르게 분류할 수 있음
# sns.lmplot(data = df2_sample, x = "(혈청지오티)AST", y = "음주여부", robust = True) # robust 옵션을 통해 이상치를 제외한 그래프를 그릴 수 있음
sns.scatterplot(data = df2_sample, x = "(혈청지오티)AST", y = "(혈청지오티)ALT", hue = "음주여부", size = "체중(5Kg 단위)") #scatterplot에서는 size 옵션을 통해 특정 번셰에 따라 크기를 다르게 분류할 수 있음

df2_chol = df2.loc[df2["총 콜레스테롤"].notnull(), "총 콜레스테롤"]
print(df2_chol.head())
sns.distplot(df2_chol) # displot는 scatterplot과 달리 한 범례에 대한 도수분포료를 바탕으로 히스토그램을 그림
sns.distplot(df2_chol, bins=10) # bin으로 데이터를 나눠 도수분포료를 작성할지 설정할 수 있음l
sns.distplot(df2.loc[(df2["총 콜레스테롤"].notnull()) & (df2["음주여부"] == 1), "총 콜레스테롤"])
sns.distplot(df2.loc[(df2["총 콜레스테롤"].notnull()) & (df2["음주여부"] == 0), "총 콜레스테롤"]) #이때 displot에는 시리즈 데이터가 들어가야함
# sns.distplot(df2.loc[(df2["총 콜레스테롤"].notnull()) & (df2["음주여부"] == 1), "총 콜레스테롤"], his = False) # hist = False 시 확률밀도함수만 그림, 이를 통해 두 개의 그래프를 같이 그릴 수 잇음
sns.kdeplot(df2.loc[(df2["총 콜레스테롤"].notnull()) & (df2["음주여부"] == 1), "총 콜레스테롤"]) # 위와 동일한 역할을 함
sns.kdeplot(df2.loc[(df2["총 콜레스테롤"].notnull()) & (df2["음주여부"] == 1), "총 콜레스테롤"], label = "음주 중")
sns.kdeplot(df2.loc[(df2["총 콜레스테롤"].notnull()) & (df2["음주여부"] == 1), "총 콜레스테롤"], label = "음주 안 함") # label을 통해 각 데이터를 설명할 수 있음
plt.axvline(df2_sample["총 콜레스테롤"].mean(), linestyle=":")
plt.axvline(df2_sample["총 콜레스테롤"].median(), linestyle="--") # plt.axvline을 통해 평균갑과 중앙값을 기호로 나타낼 수 있음

# 상관계수로 가설검정하기
# df2_small = df2_sample.columns
# df2_corr = df2_small.corr()
# df2_corr["신장(5Cm단위)"].sort_values() #신장에 대한 다른 범례의 상관계수를 정렬한 값을 볼 수 있음
# df2_corr.loc[df2_corr["신장(5Cm단위)"] > 0.3, "신장(5Cm단위)"] #신장에 대한 상관계수가 0.3 이상인 범례만 가져옴
# df2_corr.loc[df2_corr["음주여부"] > 0.25, "음주여부"] #음주여부와의 상관관계가 0.25 이상인 범례만 가져옴
# df2_corr["혈색소"].sort_values(ascending=False).head(7)
# df2_corr["감마지티피"].sort_values(ascending=False).head(7) # sort_values를 통해 의미있는 상관계수를 가진 범례를 확인할 수 있음
#
# plt.figure(figsize = (20,7))
# sns.heatmap(df2_corr, annot = True, fmt = ".2f", cmap = "Blues") # heatmap은 데이터 간의 상관관계에 따라 다른 색으로 표시해주는 도표를 그려줌
# mask = np.triu(np.ones_like(df2_sample.corr(), dtype=np.bool)) # 대각선 아래만 표시되는 numpy 설정을 가져올 수 있음
# sns.heatmap(df2_corr, annot = True, fmt = ".2f", cmap = "Blues", mask = mask) # mask에 가져온 설정을 대입하면 대각선 아래에만 표시되는 도표를 얻을  수 잇음



#실제 데이터 본격적으로 분석해보기
if os.name == "nt":
    sns.set(font="Malgun Gothic") #글꼴 설정

df3 = pd.read_csv("data/국가_대륙_별_상품군별_온라인쇼핑_해외직접판매액_20210716133340.csv", encoding = "euc-kr")
print(df3.shape)
print(df3["국가(대륙)별"].value_counts())
print(df3[df3["국가(대륙)별"] == "미국"]) #데이터 살펴보기
df3.melt(id_vars = ["국가(대륙)별", "상품군별", "판매유형별"]) #데이터를 tidy data 형식으로 만들어야 분석에 편함, 이때 왼쪽에서 행을 구분하는 역할을 하는 id 범례들을 id_vars에 담을 수 있음
df3 = df3.melt(id_vars = ["국가(대륙)별", "상품군별", "판매유형별"], var_name = "기간", value_name = "백만원") # id에 포함되지 않은 범례는 id 오른쪽에서 variable이 되어 value값을 지정해주는 역할을 함
print(df3.head()) # 값에 "합계"가 많으므로 전처리를 통해 정리해줘야함                                           # melt시에 var와 value의 이름을 바꿔줄 수 있음
print(df3.info()) # 연도와 분기를 분리하고 object가 아닌 숫자로 타입을 변경해야함

"2019 1/4 p)".split() # split은 공백을 기준으로 데이터를 분리함
"2019 1/4 p)".split()[0] #첫 번째 인덱스만 골라 분리할 수 있음
int("2019 1/4 p)".split()[0]) #분리한 값을 int형으로 바꿔줌
df3["기간"].map(lambda x : int (x.split()[0])) # map은 안에 함수를 보관하고 꺼내쓸 수 있음, 첫번째 인덱스를 분리해 int로 바꿔주는 함수를 보관했음
df3["연도"] = df3["기간"].map(lambda x : int (x.split()[0])) # 이를 새로운 열인 연도에 대입해줌
"2019 4/4 p)".split()[1].split("/")[0] # 공백을 기준으로 분리하고 다시 /을 기준으로 분리한 후 원하는 인덱스를 가져왔음
df3["분기"] = df3["기간"].map(lambda x : int(x.split()[1]).split("/")[0]) # 이를 연도와 같은 방법으로 새로운 열인 분기에 넣어줌
df3["백만원"].replace("-", pd.np.nan) #결측치는 -가 아니라 NaN으로 대체해줌
df["백만원"] = df["백만원"].replace("-", pd.np.nan).astype(float) #대체된 값을 astype을 통해 실수로 바꿔주어 다시 백만원 열에 대입했음
df3 = df3[(df3["국가(대륙)별"] != "합계") & (df3["상품군별"] != "합계")].copy() #"합계" 값이 너무 많으므로 아닌 부분만 집계해 다시 df3에 대힙했음
print(df3.sample(100, random_state = True))

#전체 상품 판매추이 파악해보기
df3_total = df3[df3["판매유형별"] == "계"] #NaN 값을 포합하지 않도록 판매유형별이 '계'인 데이터만 가져옴
sns.lineplot(data = df3_total, x = "연도", y = "백만원") #전체 판매액을 lineplot으로 그려보았음
sns.lineplot(data = df3_total, x = "연도", y = "백만원", hue = "상품군별") #상품군 별 판매액을 lineplot으로 그려보았음
plt.legend(bbox_to_anchor = (1.05,1), loc = 2, borderaxespad = 0.) #색상을 상품마다 다르게 설정
sns.relplot(data = df3_total, x = "연도", y = "백만원", hue = "상품군별", kind = "line") #한번에 그리기
sns.relplot(data = df3_total, x = "연도", y = "백만원", hue = "상품군별", kind = "line", col = "상품군별", col_wrap = 4) #한 행에 4개씩 나눠서 그리기
df3_sub = df3_total[~df3_total["상품군별"].isin(["화장품"])].copy() # isin을 통해 증가 폭이 큰 화장품을 제외시킴
sns.relplot(data = df3_sub, x = "연도", y = "백만원", hue = "상품군별", col = "상품군별", col_wrap = 4, kind = "line")
df3_sub = df3_total[~df3_total["상품군별"].isin(["화장품", "의류 및 패션관련 상품"])].copy() #증가폭이 큰 의류 및 패션상품 또한 제외시킴
sns.relplot(data = df3_sub, x = "연도", y = "백만원", hue = "상품군별", col = "상품군별", col_wrap = 4, kind = "line") #가전과 음반의 판매가 두드러짐을 알 수 있었음

#화장품 판매추이 파악해보기
df3_cosmetic = df3_total[df3_total["상품군별"] == "화장품"].copy()
sns.lineplot(data = df3_cosmetic, x = "연도", y = "백만원") #연도별 화장품 판매액을 lineplot으로 그려보았음
plt.figure(figsize = (15, 4))
sns.lineplot(data = df3_cosmetic, x = "연도", y = "백만원", hue = "분기") #분기별로 보아도 비슷함
plt.xticks(rotation = 30) #가독성위해 글자 회전
sns.lineplot(data = df3_cosmetic, x = "기간", y = "백만원") #전체 기간으로 보아도 비슷함
print(df3_cosmetic.head()) #국가대륙별로 출력하기 위해 DF 확인
plt.xticks(rotation = 30)
sns.lineplot(data = df3_cosmetic, x = "기간", y = "백만원", hue = "국가(대륙)별") #중국의 판매액이 가장 높음
plt.xticks(rotation=30)
sns.lineplot(data = df3_cosmetic[df3_cosmetic["국가(대륙별)"] != "중국"], x = "기간", y = "백만원", hue = "국가(대륙)별") #중국 제외시 아세안이 가장 높음
plt.xticks(rotation=30)
df3_sub = df3[df3["판매유형별"] != "계"].copy() #판매유형별 판매추이를 보기위해 계 데이터를 제외
sns.lineplot(data = df3_sub, x = "기간", y = "백만원", hue = "판매유형별") #면세점과 비면세점의 판매추이를 lineplot으로 그려보았음
df3_sub = df3[(df3["판매유형별"] != "계") & (df3["판매유형별"] != "면세점")].copy()
sns.lineplot(data = df3_sub, x = "기간", y = "백만원", hue = "판매유형별", ci = None) #비면세점에서도 추이는 비슷함

#패션의류 판매추이 파악해보기
df3_fashion = df3[df3["상품군별"] == "의류 및 패션관련 상품"].copy()
df3_fashion = df3[(df3["상품군별"] == "의류 및 패션관련 상품") & (df3["판매유형별"] == "계")].copy()
plt.xticks(rotation=30)
sns.lineplot(data = df3_fashion, x = "기간", y = "백만원", hue = "국가(대륙)별") #국가대륙별 전체 패션의류 판매추이를 lineplot을 통해 그려보았음
df3_fashion2 = df3[(df3["상품군별"] == "의류 및 패션관련 상품") & (df3["판매유형별"] != "계")].copy()
plt.xticks(rotation = 30)
sns.lineplot(data = df3_fashion2, x = "기간", y = "백만원", hue = "판매유형별", ci = None) #면세점과 비면세점의 판매추이를 lineplot으로 그려보았음
df3_fashion = df3_fashion.pivot_table(index = "국가(대륙)별", columns = "연도", values = "백만원") # index로 행을 구분하고 columns로 열을 구분하여 values에서 값을 찾아 도표를 그림, aggfunc를 지정하지 않으면 기본적으로 평균으로 표시함
df3_fashion_res = df3_fashion.pivot_table(index = "국가(대륙)별", columns = "연도", values = "백만원", aggfunc = "sum")
sns.heatmap(df3_fashion_res) # heatmap을 통해 값의 추이를 색상으로 나타내주는 도표를 그림
sns.heatmap(df3_fashion_res, cmap = "Blues_r") # cmap 옵션을 통해 파란색으로 색깔을 바굼
sns.heatmap(df3_fashion_res, cmap = "Blues_r", annot = True) # annot 옵션을 통해 수치를 표시해줌
sns.heatmap(df3_fashion_res, cmap = "Blues_r", annot = True, fmt = ".0f") # fmt옵션을 통해숫자를 실수형태로 표시하고, 소숫점 자릿수를 바꿀 수 있음)