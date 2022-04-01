import pandas as pd
import folium as fl
import json as js

# 데이터 전처리
# 병원자료
dataframe = pd.read_csv("data/Beta2021/1. 병원정보서비스 2021.6.csv", encoding = "cp949")
# 공공데이터포털의 '건강보험심사평가원_전국 병의원 및 약국 현황' 자료에서 2021년 6월 시점의 병원정보서비스 데이터를 활용
print(dataframe.isnull().sum())
dataframe = dataframe.dropna(subset = ["x좌표", "y좌표"]) # X 좌표나 Y 좌표에 결측치가 있는 병원 제거
drop_hospitals = dataframe.loc[dataframe["총의사수"] < 11, "총의사수"].index
drop_hospitals = drop_hospitals.tolist()
dataframe = dataframe.drop(drop_hospitals, axis = 0) # 의사수가 11명 미만인 병원 제거
dataframe = dataframe.loc[dataframe["종별코드명"].isin(["상급종합", "종합병원", "병원"])].copy() # 1차 병원 및 보건소 제거

dataframe = dataframe.drop("암호화YKIHO코드", axis = "columns")
dataframe = dataframe.drop("우편번호", axis = "columns")
dataframe = dataframe.drop("전화번호", axis = "columns")
dataframe = dataframe.drop("병원URL", axis = "columns")
dataframe = dataframe.drop("읍면동", axis = "columns")
dataframe = dataframe.drop("개설일자", axis = "columns")
dataframe = dataframe.drop("종별코드", axis = "columns")
dataframe = dataframe.drop("시도코드", axis = "columns")
dataframe = dataframe.drop("시군구코드", axis = "columns")
dataframe = dataframe.drop("의과일반의 인원수", axis = "columns")
dataframe = dataframe.drop("의과인턴 인원수", axis = "columns")
dataframe = dataframe.drop("의과레지던트 인원수", axis = "columns")
dataframe = dataframe.drop("의과전문의 인원수", axis = "columns")
dataframe = dataframe.drop("치과일반의 인원수", axis = "columns")
dataframe = dataframe.drop("치과인턴 인원수", axis = "columns")
dataframe = dataframe.drop("치과레지던트 인원수", axis = "columns")
dataframe = dataframe.drop("치과전문의 인원수", axis = "columns")
dataframe = dataframe.drop("한방일반의 인원수", axis = "columns")
dataframe = dataframe.drop("한방인턴 인원수", axis = "columns")
dataframe = dataframe.drop("한방레지던트 인원수", axis = "columns")
dataframe = dataframe.drop("한방전문의 인원수", axis = "columns") # 필요없는 컬럼을 제거

dataframe = dataframe.loc[dataframe["시도코드명"].isin(["서울", "인천", "경기"])].copy() # 수도권의 병원들만 뽑아옴
dataframe_ICN = dataframe.loc[dataframe["시도코드명"].isin(["인천"])].copy() # 인천의 병원들만 뽑아옴
dataframe_SEL = dataframe.loc[dataframe["시도코드명"].isin(["서울"])].copy() # 서울의 병원들만 뽑아옴
dataframe_GG = dataframe.loc[dataframe["시도코드명"].isin(["경기"])].copy() # 경기의 병원들만 뽑아옴
dataframe_GG_North = dataframe_GG.loc[dataframe_GG["시군구코드명"].isin(["고양덕양구", "고양일산동구", "고양일산서구", "구리시", "남양주시", "동두천시", "양주시", "의정부시", "파주시", "가평군", "연천군", "포천시"])].copy() # 경기북부의 병원들만 뽑아옴
drop_cities = dataframe_GG.loc[dataframe_GG["시군구코드명"].isin(["고양덕양구", "고양일산동구", "고양일산서구", "구리시", "남양주시", "동두천시", "양주시", "의정부시", "파주시", "가평군", "연천군", "포천시"])].index
dataframe_GG_South = dataframe_GG.drop(drop_cities, axis = 0) # 경기남부의 병원들만 뽑아옴

# 인구 자료
dataframe_Population = pd.read_csv("data/Beta2021/행정구역_읍면동_별_5세별_주민등록인구_2011년__20210810183451.csv", encoding = "cp949")
# 국가통계포털의 '행정구역(읍면동)별/5세별 주민등록인구(2011년~)' 자료에서 2021년 8월 10일 시점의 전국 시군 및 구(광역시) 단위의 인구데이터를 활용
print(dataframe_Population.isnull().sum())
drop_gu = dataframe_Population.loc[dataframe_Population["행정구역(동읍면)별"].isin(["수원시", "성남시", "용인시", "고양시", "안양시", "안산시", "서울특별시", "경기도", "인천광역시"])].index # 구가 존재하는 시들의 인구 합계 데이터를 제외
dataframe_Population = dataframe_Population.drop(drop_gu, axis = 0)
dataframe_Population = dataframe_Population.drop("항목", axis = 1)
dataframe_Population = dataframe_Population.drop("0 - 4세", axis = 1)
dataframe_Population = dataframe_Population.drop("5 - 9세", axis = 1)
dataframe_Population = dataframe_Population.drop("10 - 14세", axis = 1)
dataframe_Population = dataframe_Population.drop("15 - 19세", axis = 1)
dataframe_Population = dataframe_Population.drop("20 - 24세", axis = 1)
dataframe_Population = dataframe_Population.drop("25 - 29세", axis = 1)
dataframe_Population = dataframe_Population.drop("30 - 34세", axis = 1)
dataframe_Population = dataframe_Population.drop("35 - 39세", axis = 1)
dataframe_Population = dataframe_Population.drop("40 - 44세", axis = 1)
dataframe_Population = dataframe_Population.drop("45 - 49세", axis = 1)
dataframe_Population = dataframe_Population.drop("50 - 54세", axis = 1)
dataframe_Population = dataframe_Population.drop("55 - 59세", axis = 1) # 필요없는 컬럼을 제거

dataframe = dataframe.reset_index()
dataframe_Population = dataframe_Population.reset_index()
dataframe_Population["지역별 총의사수"] = 0
for X in range(len(dataframe)) :
    for Y in range(len(dataframe_Population)) :
        if dataframe.loc[X, "시군구코드명"] == dataframe_Population.loc[Y, "행정구역(동읍면)별"] :
            dataframe_Population.loc[Y, "지역별 총의사수"] += dataframe.loc[X, "총의사수"]
            break
        else :
            continue # 병원 별 총의사수를 이용해 지역별 총의사수를 구함
for Z in range(len(dataframe_Population)) :
    if dataframe_Population.loc[Z, "지역별 총의사수"] == 0 :
        dataframe_Population.loc[Z ,"지역별 의사당 담당 환자수"] = 4287
    else :
        dataframe_Population.loc[Z, "지역별 의사당 담당 환자수"] = dataframe_Population.loc[Z, "60 - 100+"] / dataframe_Population.loc[Z, "지역별 총의사수"]
# 지역별 총의사수를 이용해 지역별 의사당 담당 환자수를 구함, 의사가 없는 지역은 의사당 담당 환자수 데이터의 최대값으로 설정

dataframe_Population_SEL = dataframe_Population.loc[dataframe_Population["행정구역(동읍면)별"].isin(["종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "도봉구", "노원구",
"은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구"])].copy() # 서울 행정구역들의 인구 자료를 불러옴
dataframe_Population_ICN = dataframe_Population.loc[dataframe_Population["행정구역(동읍면)별"].isin(["인천중구", "인천동구", "인천미추홀구", "인천연수구", "인천남동구", "인천부평구", "인천계양구",
"인천서구", "인천강화군", "인천옹진군"])].copy() # 인천 행정구역들의 인구 자료를 불러옴
dataframe_Population_GG_North = dataframe_Population.loc[dataframe_Population["행정구역(동읍면)별"].isin(["고양덕양구", "고양일산동구", "고양일산서구", "구리시", "남양주시", "동두천시", "양주시", "의정부시",
"파주시", "가평군", "연천군", "포천시"])].copy() # 경기 북부 행정구역들의 인구 자료를 불러옴
dataframe_Population_GG_South = dataframe_Population.loc[dataframe_Population["행정구역(동읍면)별"].isin(["수원장안구", "수원권선구", "수원팔달구", "수원영통구", "성남수정구", "성남중원구", "성남분당구", "안양만안구",
"안양동안구", "부천시", "광명시", "평택시", "안산상록구", "안산단원구", "과천시", "오산시", "시흥시", "군포시", "의왕시", "하남시", "용인처인구", "용인기흥구", "용인수지구", "이천시", "안성시", "김포시", "화성시",
"광주시", "여주시", "양평군"])].copy() # 경기 남부 행정구역들의 인구 자료를 불러옴

# 데이터 시각화
geo_data = js.load(open("data/Beta2021/skorea_municipalities_geo_simple.json", encoding = "utf-8")) # 전국 시군 및 구(광역시) 지리 데이터를 불러옴
Map1_Hospital = fl.Map(location = [dataframe_ICN["y좌표"].mean(), dataframe_ICN["x좌표"].mean()], zoom_start = 11)
for X in dataframe_ICN.index :
    if dataframe_ICN["총의사수"][X] > 1000 :
        Radius = 25
    elif (dataframe_ICN["총의사수"][X] > 500) :
        Radius = 18
    elif (dataframe_ICN["총의사수"][X] > 300) :
        Radius = 12
    elif (dataframe_ICN["총의사수"][X] > 100) :
        Radius = 10
    elif (dataframe_ICN["총의사수"][X] > 50) :
        Radius = 8
    elif (dataframe_ICN["총의사수"][X] > 20) :
        Radius = 7
    else :
        Radius = 6
    fl.CircleMarker(
        location = [dataframe_ICN["y좌표"][X], dataframe_ICN["x좌표"][X]],
        radius = Radius,
        weight = 1.5,
        color = "#752671",
        fill_color = "#D9E600",
        fill_opacity = 0.5
    ).add_to(Map1_Hospital)
Map1_Hospital.save("Map1_Hospital.html") # 인천 전체 병원 분포를 시각화
Map1_Population = fl.Map(location = [dataframe_ICN["y좌표"].mean(), dataframe_ICN["x좌표"].mean()], zoom_start = 11)
fl.Choropleth(
    geo_data = geo_data,
    data = dataframe_Population_ICN,
    columns = ["행정구역(동읍면)별", "지역별 의사당 담당 환자수"],
    key_on = "feature.properties.name",
    fill_color = "YlOrBr",
    fill_opacity = 0.8
).add_to(Map1_Population)
Map1_Population.save("Map1_Population.html") # 인천 행정 구역별 의사당 담당 환자수를 시각화
Map2_Hospital = fl.Map(location = [dataframe_SEL["y좌표"].mean(), dataframe_SEL["x좌표"].mean()], zoom_start = 12)
for X in dataframe_SEL.index :
    if dataframe_SEL["총의사수"][X] > 1000 :
        Radius = 25
    elif (dataframe_SEL["총의사수"][X] > 500) :
        Radius = 18
    elif (dataframe_SEL["총의사수"][X] > 300) :
        Radius = 12
    elif (dataframe_SEL["총의사수"][X] > 100) :
        Radius = 10
    elif (dataframe_SEL["총의사수"][X] > 50) :
        Radius = 8
    elif (dataframe_SEL["총의사수"][X] > 20) :
        Radius = 7
    else :
        Radius = 6
    fl.CircleMarker(
        location = [dataframe_SEL["y좌표"][X], dataframe_SEL["x좌표"][X]],
        radius = Radius,
        weight = 1.5,
        color = "#752671",
        fill_color = "#D9E600",
        fill_opacity = 0.5
    ).add_to(Map2_Hospital)
Map2_Hospital.save("Map2_Hospital.html") # 서울 전체 병원 분포를 시각화
Map2_Population = fl.Map(location = [dataframe_SEL["y좌표"].mean(), dataframe_SEL["x좌표"].mean()], zoom_start = 12)
fl.Choropleth(
    geo_data = geo_data,
    data = dataframe_Population_SEL,
    columns = ["행정구역(동읍면)별", "지역별 의사당 담당 환자수"],
    key_on = "feature.properties.name",
    fill_color = "YlOrBr",
    fill_opacity = 0.8
).add_to(Map2_Population)
Map2_Population.save("Map2_Population.html") # 서울 행정 구역별 의사당 담당 환자수를 시각화
Map3_Hospital = fl.Map(location = [dataframe_GG_North["y좌표"].mean(), dataframe_GG_North["x좌표"].mean()], zoom_start = 10)
for X in dataframe_GG_North.index :
    if dataframe_GG_North["총의사수"][X] > 1000 :
        Radius = 25
    elif (dataframe_GG_North["총의사수"][X] > 500) :
        Radius = 18
    elif (dataframe_GG_North["총의사수"][X] > 300) :
        Radius = 12
    elif (dataframe_GG_North["총의사수"][X] > 100) :
        Radius = 10
    elif (dataframe_GG_North["총의사수"][X] > 50) :
        Radius = 8
    elif (dataframe_GG_North["총의사수"][X] > 20) :
        Radius = 7
    else :
        Radius = 6
    fl.CircleMarker(
        location = [dataframe_GG_North["y좌표"][X], dataframe_GG_North["x좌표"][X]],
        radius = Radius,
        weight = 1.5,
        color = "#752671",
        fill_color = "#D9E600",
        fill_opacity = 0.5
    ).add_to(Map3_Hospital)
Map3_Hospital.save("Map3_Hospital.html") # 경기 북부 전체 병원 분포를 시각화
Map3_Popultaion = fl.Map(location = [dataframe_GG_North["y좌표"].mean(), dataframe_GG_North["x좌표"].mean()], zoom_start = 10)
fl.Choropleth(
    geo_data = geo_data,
    data = dataframe_Population_GG_North,
    columns = ["행정구역(동읍면)별", "지역별 의사당 담당 환자수"],
    key_on = "feature.properties.name",
    fill_color = "YlOrBr",
    fill_opacity = 0.8
).add_to(Map3_Popultaion) # 경기 북부 행정 구역별 의사당 담당 환자수를 시각화
Map3_Popultaion.save("Map3_Popultaion.html")
Map4_Hospital = fl.Map(location = [dataframe_GG_South["y좌표"].mean(), dataframe_GG_South["x좌표"].mean()], zoom_start = 10.4)
for X in dataframe_GG_South.index :
    if dataframe_GG_South["총의사수"][X] > 1000 :
        Radius = 25
    elif (dataframe_GG_South["총의사수"][X] > 500) :
        Radius = 18
    elif (dataframe_GG_South["총의사수"][X] > 300) :
        Radius = 12
    elif (dataframe_GG_South["총의사수"][X] > 100) :
        Radius = 10
    elif (dataframe_GG_South["총의사수"][X] > 50) :
        Radius = 8
    elif (dataframe_GG_South["총의사수"][X] > 20) :
        Radius = 7
    else :
        Radius = 6
    fl.CircleMarker(
        location = [dataframe_GG_South["y좌표"][X], dataframe_GG_South["x좌표"][X]],
        radius = Radius,
        weight = 1.5,
        color = "#752671",
        fill_color = "#D9E600",
        fill_opacity = 0.5
    ).add_to(Map4_Hospital)
Map4_Hospital.save("Map4_Hospital.html") # 경기 남부 전체 병원 분포를 시각화
Map4_Population = fl.Map(location = [dataframe_GG_South["y좌표"].mean(), dataframe_GG_South["x좌표"].mean()], zoom_start = 10.4)
fl.Choropleth(
    geo_data = geo_data,
    data = dataframe_Population_GG_South,
    columns = ["행정구역(동읍면)별", "지역별 의사당 담당 환자수"],
    key_on = "feature.properties.name",
    fill_color = "YlOrBr",
    fill_opacity = 0.8
).add_to(Map4_Population)
Map4_Population.save("Map4_Population.html") # 경기 남부 행정 구역별 의사당 담당 환자수를 시각화