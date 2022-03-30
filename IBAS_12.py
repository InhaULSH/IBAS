import pandas as pd

dataframe = pd.read_csv("../../Downloads/module_class.csv", encoding = "cp949")
print(dataframe.info())
dataframe.drop([3], axis = 0, inplace = True)
dataframe.dropna(axis = 0, inplace = True)
dataframe_A = dataframe.drop(["코딩"], axis = 1)
dataframe_B = dataframe


score_list = []
for i in range(len(dataframe_A)):
    korean_score = dataframe_A.iloc[i:i+1, 1].squeeze()
    math_score = dataframe_A.iloc[i:i+1, 2].squeeze()
    english_score = dataframe_A.iloc[i:i+1, 3].squeeze()
    all_score = korean_score + math_score + english_score
    score_list.append(all_score)
    pass
dataframe_A["총점"] = score_list
dataframe_A['평균점수'] = (dataframe_A["총점"] / 3).round(1)
dataframe_A.sort_values("평균점수", ascending = False, inplace = True)
print(dataframe_A.head())

rank_list = []
for i in range(1, 22) :
    rank_list.append(i)
    pass
dataframe_A.loc[:, '순위'] = rank_list
dataframe_A.set_index('이름', inplace = True)
print(dataframe_A)


dataframe_B["총점"] = dataframe_B["국어"] + dataframe_B["수학"] + dataframe_B["영어"] + dataframe_B["코딩"]
dataframe_B["평균점수"] = (dataframe_B["총점"] / 4).round(1)
print(dataframe_B.head())
extra_data = pd.DataFrame({'이름' : ['짱구', '훈이', '맹구', '유리'],
                          '국어' : [80, 40, 30, 50],
                          '수학' : [50, 70, 90, 30],
                          '영어' : [60, 80, 10, 20],
                          '코딩' : [20, 40, 50, 60]})
extra_data['총점'] = extra_data['국어'] + extra_data['수학'] + extra_data['영어'] + extra_data['코딩']
extra_data['평균점수'] = (extra_data['총점'] / 4).round(1)
dataframe_C = pd.concat([dataframe_B, extra_data], axis = 0)
dataframe_C.sort_values('평균점수', ascending = False, inplace = True)
dataframe_C.reset_index(drop = True, inplace = True)
rank_list_C = []
for i in range(1, 26) :
    rank_list_C.append(i)
    pass
rank_data = pd.DataFrame([rank_list_C])
rank_data = rank_data.T
rank_data.rename(columns = {0 : '순위'}, inplace = True)
print(rank_data)
dataframe_Final = pd.concat([dataframe_C, rank_data], axis = 1)
dataframe_Final.set_index('이름', inplace = True)
print(dataframe_Final)