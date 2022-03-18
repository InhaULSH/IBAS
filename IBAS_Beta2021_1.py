import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium as fl
import os

if os.name == "nt":
    sns.set(font = "Malgun Gothic")
plt.rc("axes", unicode_minus = False)

dataframe = pd.read_csv("data/한국전력공사_낙뢰관측 정보_20200413.csv", encoding = "cp949")
drop_list1 = dataframe.loc[dataframe["낙뢰크기"] == 0, "낙뢰크기"].index
drop_list1 = drop_list1.tolist()
dataframe = dataframe.drop(drop_list1, axis = 0)
drop_list2 = dataframe.loc[(dataframe["위도"] > 39), "위도"].index
drop_list2 = drop_list2.tolist()
dataframe = dataframe.drop(drop_list2, axis = 0)
drop_list3 = dataframe.loc[(dataframe["위도"] < 33), "위도"].index
drop_list3 = drop_list3.tolist()
dataframe = dataframe.drop(drop_list3, axis = 0)
drop_list4 = dataframe.loc[(dataframe["경도"] > 130), "위도"].index
drop_list4 = drop_list4.tolist()
dataframe = dataframe.drop(drop_list4, axis = 0)
drop_list5 = dataframe.loc[(dataframe["경도"] < 126), "위도"].index
drop_list5 = drop_list5.tolist()
dataframe = dataframe.drop(drop_list5, axis = 0)
dataframe["월"] = dataframe["낙뢰발생일"].map(lambda x : int (x.split("-")[1]))
dataframe["발생횟수"] = 1
for X in dataframe.index :
    if dataframe.loc[X, "낙뢰크기"] < 0 :
        dataframe.loc[X, "낙뢰크기"] = -dataframe.loc[X, "낙뢰크기"]
    else :
        continue

dataframe.groupby(["월"])["발생횟수"].count().plot.bar(figsize = (7,7))
plt.show()
dataframe.groupby(["월"])["낙뢰크기"].mean().plot.bar(figsize = (7,7))
plt.show()
map = fl.Map(location = [dataframe["위도"].mean(), dataframe["경도"].mean()], zoom_start = 6)
for X in dataframe.index :
    if (dataframe.loc[X, "월"] == 9): # 특정 값을 찾고자 하면 df.loc["행 인덱스", "열 인덱스"] 로 호줄 가능
        fillcolor = "orange"
    elif (dataframe.loc[X, "월"] == 10) :
        fillcolor = "yellow"
    elif (dataframe.loc[X, "월"] == 11) :
        fillcolor = "blue"
    else :
        fillcolor = "cadetblue"
    fl.CircleMarker(
        location = [dataframe["위도"][X], dataframe["경도"][X]],
        radius = dataframe["낙뢰크기"][X] / 8,
        weight = 1,
        color = fillcolor,
        fill_color = fillcolor,
        fill_opacity = 0.75,
        fill = True
    ).add_to(map)
map.save("map.html")