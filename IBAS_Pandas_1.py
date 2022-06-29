import pandas as pd

dataframe = pd.read_csv("./DataSet/module_class.csv", encoding = "cp949") # 학생들의 성적 파일을 불러옴
print(dataframe.info())
dataframe.drop([3], axis = 0, inplace = True)
dataframe.dropna(axis = 0, inplace = True)
dataframe_A = dataframe.drop(["코딩"], axis = 1) # 코딩 성적을 뺀 데이터 프레임
dataframe_B = dataframe # 코딩 성적을 포함한 데이터 프레임

score_list = []
for i in range(len(dataframe_A)):
    korean_score = dataframe_A.iloc[i:i+1, 1].squeeze()
    math_score = dataframe_A.iloc[i:i+1, 2].squeeze()
    english_score = dataframe_A.iloc[i:i+1, 3].squeeze()
    all_score = korean_score + math_score + english_score
    score_list.append(all_score)
    pass
# 각 학생의 국영수 점수를 뽑아 총점을 구해 총점 리스트를 만듬
dataframe_A["총점"] = score_list # 총점 리스트를 데이터프레임 A 의 새로운 열로 추가
dataframe_A['평균점수'] = (dataframe_A["총점"] / 3).round(1) # 총점으로 부터 평균점수 열을 추가
dataframe_A.sort_values("평균점수", ascending = False, inplace = True) # 평균점수 순서대로 객체(학생)들을 정렬
print(dataframe_A.head())

rank_list = []
for i in range(1, 22) :
    rank_list.append(i)
    pass
# 정렬된 데이터 프레임에서의 등수(인덱스)를 리스트로 만듬
dataframe_A.loc[:, '순위'] = rank_list # 등수 리스트를 데이터프레임 A 에 추가
dataframe_A.set_index('이름', inplace = True) # 이름을 인덱스로 사용
print(dataframe_A)


dataframe_B["총점"] = dataframe_B["국어"] + dataframe_B["수학"] + dataframe_B["영어"] + dataframe_B["코딩"]
dataframe_B["평균점수"] = (dataframe_B["총점"] / 4).round(1) # 데이터 프레임 B 에서 총점과 평균점수 열을 추가
print(dataframe_B.head())
extra_data = pd.DataFrame({'이름' : ['짱구', '훈이', '맹구', '유리'],
                          '국어' : [80, 40, 30, 50],
                          '수학' : [50, 70, 90, 30],
                          '영어' : [60, 80, 10, 20],
                          '코딩' : [20, 40, 50, 60]})
extra_data['총점'] = extra_data['국어'] + extra_data['수학'] + extra_data['영어'] + extra_data['코딩']
extra_data['평균점수'] = (extra_data['총점'] / 4).round(1) # 동일한 형태의 새 데이터 프레임 선언
dataframe_C = pd.concat([dataframe_B, extra_data], axis = 0) # 동일한 형태의 두 데이터 프레임 병합
dataframe_C.sort_values('평균점수', ascending = False, inplace = True) # 병합된 데이터 프레임 C 를 평균점수따라 정렬
dataframe_C.reset_index(drop = True, inplace = True)
rank_list_C = []
for i in range(1, 26) :
    rank_list_C.append(i)
    pass
rank_data = pd.DataFrame([rank_list_C]) # 등수 리스트를 바탕으로 등수 데이터프레임 선언
rank_data = rank_data.T # 행과 열을 교체
rank_data.rename(columns = {0 : '순위'}, inplace = True) # 열의 이름을 바꿈
print(rank_data)
dataframe_Final = pd.concat([dataframe_C, rank_data], axis = 1) # 등수 데이터 프레임과 데이터 프레임 C 를 병합
dataframe_Final.set_index('이름', inplace = True) # 이름을 인덱스로 사용
print(dataframe_Final)