# 코스피 주요 종목 월별 종가 데이터 불러오기 (2017 / 12 ~ 2022 / 11, 시가총액은 2022 / 12 / 21 기준)
# KRX 정보 데이터 시스템 - 기본통계 - 주식 - 종목시세 - 종목시세 추이(월/연도) 에서 연월, 종목명, 종목코드, 종가 열을 CSV 파일로 불러옴
# 일별 데이터가 아닌 월별 데이터를 사용하는 대신 논문에서 제시된 표본보다 많은 85개의 종목을 선택, 연산성능 부담은 줄어드나 모델의 실시간성은 떨어짐
# 시가총액 상위 85개 종목을 선택, 5개년치 종가 데이터가 존재하지 않거나 우선주이거나 민간기업의 지주회사(주요 자회사가 상장되있는 경우) 는 제외
# 선택된 종목별 파일을 ICT, ENGINEERING, BIO & BEAUTY, RETAIL, TRANSPORT, FINANCE, PUBLIC, SERVICE 분야로 분류하여 폴더로 정리
import pandas as pd
import numpy as np
import os

""" 
def mergeCSV(file_path, file_format, save_path, save_format, columns=None):
    merge_df = pd.DataFrame()
    file_list = file_list = [f"{file_path}/{file}" for file in os.listdir(file_path) if file_format in file]

    for file in file_list:
        file_df = pd.read_csv(file, encoding = "cp949")
        if columns is None:
            columns = file_df.columns

        temp_df = pd.DataFrame(file_df, columns=columns)
        merge_df = merge_df.append(temp_df)
    else:
        merge_df.to_csv(save_path, index=False, encoding = "cp949")

mergeCSV(file_path = './DataSet_Beta2022/ICT', file_format = '.csv', save_path = './Csv_Beta2022/ICT.csv', save_format = '.csv')
mergeCSV(file_path = './DataSet_Beta2022/PUBLIC', file_format = '.csv', save_path = './Csv_Beta2022/PUBLIC.csv', save_format = '.csv')
mergeCSV(file_path = './DataSet_Beta2022/SERVICE', file_format = '.csv', save_path = './Csv_Beta2022/SERVICE.csv', save_format = '.csv')
mergeCSV(file_path = './DataSet_Beta2022/TRANSPORT', file_format = '.csv', save_path = './Csv_Beta2022/TRANSPORT.csv', save_format = '.csv')
mergeCSV(file_path = './DataSet_Beta2022/BIO & BEAUTY', file_format = '.csv', save_path = './Csv_Beta2022/BIO & BEAUTY.csv', save_format = '.csv')
mergeCSV(file_path = './DataSet_Beta2022/ENGINEERING', file_format = '.csv', save_path = './Csv_Beta2022/ENGINEERING.csv', save_format = '.csv')
mergeCSV(file_path = './DataSet_Beta2022/FINANCE', file_format = '.csv', save_path = './Csv_Beta2022/FINANCE.csv', save_format = '.csv')
mergeCSV(file_path = './Csv_Beta2022', file_format = '.csv', save_path = './Csv_Beta2022/ALL.csv', save_format = '.csv') 
"""
# Csv_Beta2022 폴더에 분야별 종가 CSV 파일 및 전체 종목 종가 데이터 CSV 만듦


# 분야별 종가 CSV 파일 및 전체 종목 종가 데이터 CSV 전처리
