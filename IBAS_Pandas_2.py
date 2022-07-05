import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname = "C:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family = font_name) # 시각화에서 한글폰트를 쓰기위해 필요
import seaborn as sbn

dataframe = pd.read_csv('./DataSet/Iris.csv', sep = ',', encoding = 'cp949', index_col = 0) # 붓꽃 데이터를 불러옴
# 특정 인덱스의 열을 행의 인덱스로 할 수 있음
dataframe.info() # 붓꽃 자료에 어떤 변수가 있는지, 객체가 얼마나 있는지 확인
print(dataframe['Species'].value_counts(normalize=True)) # 어떤 종의 붓꽃이 있는지 확인
dataframe = dataframe.rename(columns = {'SepalLengthCm' : 'LengthOfSepal', 'SepalWidthCm' : 'WidthOfSepal',
                            'PetalLengthCm' : 'LengthOfPetal', 'PetalWidthCm' : 'WidthOfPetal'}) # 변수의 이름 변경
sbn.boxplot(x = 'Species', y = 'LengthOfSepal', data = dataframe) # 종 별로 Sepal의 길이를 상자그림으로 시각화
sbn.boxplot(x = 'Species', y = 'WidthOfSepal', data = dataframe) # 종 별로 Sepal의 넓이를 상자그림으로 시각화
dataframe.groupby('Species').sum() # 종 별로 기술통계량을 표시
sbn.pairplot(data = dataframe, size = 3, hue = 'Species') # 변수별로 같은 변수끼리는 해당 변수의 종에 따른 분포도로, 다른 변수끼리는 두 변수 끼리의산점도로 시각화
# 길이1 분    산    산    산
# 넓이1 산    분    산    산
# 길이2 산    산    분    산
# 넓이2 산    산    산    분
#     길이1 넓이1 길이2 넓이2
sbn.swarmplot(x = 'Species', y = 'LengthOfSepal', data = dataframe) # 한 변수에서 종에 따른 분포도를 시각화, 점 그래프 형태
sbn.violinplot(x = 'Species', y = 'LengthOfSepal', data = dataframe) # 한 변수에서 종에 따른 분포도를 시각화, 바이올린 그래프 형태
sbn.heatmap(dataframe.corr(), annot = True, cmap = 'Blues') # 변수끼리의 상관계수를 히트맵으로 시각화

dataframe_score = pd.read_csv('./DataSet/student_score.csv', sep = ',', encoding = 'cp949') # 학생점수 데이터를 불러옴
print(dataframe_score)
x = dataframe_score['이름']
y = dataframe_score['평균 점수']
plt.plot(x, y) # 학생별 평균 점수를 꺾은 선 그래프로 시각화
plt.bar(x, y) # 학생별 평균 점수를 막대 그래프로 시각화
plt.ylim(40, 85) # Y축을 특정 구간으로 표시
plt.xlabel("Student's Name")
plt.ylabel("Student's Score") # X, Y축 이름을 정함
plt.title("Mean Score of Students") # 그래프 이름을 정함