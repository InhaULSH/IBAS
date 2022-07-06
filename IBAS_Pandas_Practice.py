import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)


dataframe = pd.read_csv("./DataSet/UCI_Credit_Card.csv", encoding = "cp949", index_col = 0)
# 고객의 신용카드 사용정보 불러오기
# 자세한 내용은 IBAS_Pandas_Practice.docx 참고


print(dataframe.isnull().sum())
dataframe.rename(columns = {'PAY_0' : 'REPAYMENT STATUS_SEP', 'PAY_2' : 'REPAYMENT STATUS_AUG', 'PAY_3' : 'REPAYMENT STATUS_JUL'
    , 'PAY_4' : 'REPAYMENT STATUS_JUN', 'PAY_5' : 'REPAYMENT STATUS_MAY', 'PAY_6' : 'REPAYMENT STATUS_APR'}, inplace = True)
dataframe.rename(columns = {'BILL_AMT1' : 'CARD BILL_SEP', 'BILL_AMT2' : 'CARD BILL_AUG', 'BILL_AMT3' : 'CARD BILL_JUL'
    , 'BILL_AMT4' : 'CARD BILL_JUN', 'BILL_AMT5' : 'CARD BILL_MAY', 'BILL_AMT6' : 'CARD BILL_APR'}, inplace = True)
dataframe.rename(columns = {'PAY_AMT1' : 'PREPAID_SEP', 'PAY_AMT2' : 'PREPAID_AUG', 'PAY_AMT3' : 'PREPAID_JUL'
    , 'PAY_AMT4' : 'PREPAID_JUN', 'PAY_AMT5' : 'PREPAID_MAY', 'PAY_AMT6' : 'PREPAID_APR'}, inplace = True)
dataframe.rename(columns = {'LIMIT_BAL' : 'CREDIT LIMIT', 'default.payment.next.month' : 'DEFAULT'}, inplace = True)
print(dataframe.info())
# 이상치 확인, 변수 이름 변경

dataframe.replace({'SEX' : 1}, 'Male', inplace = True)
dataframe.replace({'SEX' : 2}, 'Female', inplace = True)
dataframe.replace({'EDUCATION' : 1}, 'Graduate', inplace = True)
dataframe.replace({'EDUCATION' : 2}, 'Undergraduate', inplace = True)
dataframe.replace({'EDUCATION' : 3}, 'High School', inplace = True)
dataframe.replace({'EDUCATION' : 4}, 'Less than Middle School', inplace = True)
dataframe.replace({'EDUCATION' : 5}, 'Unknown', inplace = True)
dataframe.replace({'EDUCATION' : 6}, 'Unknown', inplace = True)
dataframe.replace({'MARRIAGE' : 1}, 'Married', inplace = True)
dataframe.replace({'MARRIAGE' : 2}, 'Single', inplace = True)
dataframe.replace({'MARRIAGE' : 3}, 'Others', inplace = True)
RepayStatVariable = ['REPAYMENT STATUS_SEP', 'REPAYMENT STATUS_AUG', 'REPAYMENT STATUS_JUL',
                     'REPAYMENT STATUS_JUN', 'REPAYMENT STATUS_MAY', 'REPAYMENT STATUS_APR']
for i in RepayStatVariable:
    dataframe.replace({i : -1}, 'No Overdue', inplace=True)
    dataframe.replace({i : 1}, 'Short-term Overdue', inplace = True)
    dataframe.replace({i : 2}, '2 Months ~ 6 Months Overdue', inplace = True)
    dataframe.replace({i : 3}, '2 Months ~ 6 Months Overdue', inplace = True)
    dataframe.replace({i : 4}, '2 Months ~ 6 Months Overdue', inplace = True)
    dataframe.replace({i : 5}, '2 Months ~ 6 Months Overdue', inplace = True)
    dataframe.replace({i : 6}, 'Long-term Overdue', inplace = True)
    dataframe.replace({i : 7}, 'Long-term Overdue', inplace=True)
    dataframe.replace({i : 8}, 'Long-term Overdue', inplace=True)
    dataframe.replace({i : 9}, 'Long-term Overdue', inplace=True)
dataframe.replace({'DEFAULT': 0}, 'Yes', inplace=True)
dataframe.replace({'DEFAULT': 1}, 'No', inplace=True)
# 각 범주 변수의 관측값 이름 변경

UnkownCustomer = dataframe.loc[(dataframe['EDUCATION'] == 'Unknown') | (dataframe['EDUCATION'] == 0) |
                               (dataframe['MARRIAGE'] == 'Others') | (dataframe['MARRIAGE'] == 0)].index
dataframe.drop(UnkownCustomer, inplace = True)
# 학력을 알수 없는 고객, 결혼상태가 Others인 고객 삭제
RepayStatVariable = ['REPAYMENT STATUS_SEP', 'REPAYMENT STATUS_AUG', 'REPAYMENT STATUS_JUL',
                     'REPAYMENT STATUS_JUN', 'REPAYMENT STATUS_MAY', 'REPAYMENT STATUS_APR']
for i in RepayStatVariable:
    dataframe.replace({i : 0}, 'No Overdue', inplace=True)
    dataframe.replace({i : -2}, 'No Overdue', inplace = True)
# 상환 상태가 범례에서 벗어난 고객은 관측값을 수정
MinersBillCustomer = dataframe.loc[(dataframe['CARD BILL_SEP'] < 0) | (dataframe['CARD BILL_AUG'] < 0) | (dataframe['CARD BILL_JUL'] < 0) |
                                   (dataframe['CARD BILL_JUN'] < 0) | (dataframe['CARD BILL_MAY'] < 0) | (dataframe['CARD BILL_APR'] < 0)].index
dataframe.drop(MinersBillCustomer, inplace = True)
# 청구액이 음수인 고객 삭제


sns.countplot(x = dataframe['SEX'])
plt.show()
print(dataframe['SEX'].value_counts())
# 표본에서 성비 시각화
sns.countplot(x = dataframe['EDUCATION'], order = ['Graduate', 'Undergraduate', 'High School', 'Less than Middle School'])
plt.show()
print(dataframe['EDUCATION'].value_counts())
# 표본에서 교육수준 시각화
sns.countplot(x = dataframe['MARRIAGE'])
plt.show()
print(dataframe['MARRIAGE'].value_counts())
# 표본에서 결혼여부 시각화
sns.histplot(x = dataframe['AGE'])
plt.show()
# 표본에서 나이의 분포 시각화

print(dataframe.groupby(['SEX']).median())
print(dataframe.groupby(['SEX']).std())
# 성별에 따른 변수 중앙값, 표준편차
Q1_s1 = dataframe['CREDIT LIMIT'].quantile(0.25)
Q3_s1 = dataframe['CREDIT LIMIT'].quantile(0.75)
IQR_s1 = Q3_s1 - Q1_s1
sns.boxplot(x = dataframe['SEX'], y = dataframe.loc[(dataframe['CREDIT LIMIT'] > Q1_s1 - 1.5 * IQR_s1)
            & (dataframe['CREDIT LIMIT'] < Q3_s1 + 1.5 * IQR_s1)]['CREDIT LIMIT'], data = dataframe)
plt.show()
# 성별에 따른 신용한도 시각화
Fig1 = plt.figure(figsize = (25, 25))
for i in range(len(RepayStatVariable)):
    Fig1_sub = Fig1.add_subplot(3, 2, i + 1)
    sns.countplot(x = dataframe['SEX'], hue = dataframe[RepayStatVariable[i]], hue_order =
    ['No Overdue', 'Short-term Overdue', '2 Months ~ 6 Months Overdue', 'Long-term Overdue'], data = dataframe)
    pass
plt.show()
# 성별에 따른 상환 상태 시각화
CardBillVariable = ['CARD BILL_SEP', 'CARD BILL_AUG', 'CARD BILL_JUL', 'CARD BILL_JUN', 'CARD BILL_MAY', 'CARD BILL_APR']
Fig2 = plt.figure(figsize = (25, 25))
for i in range(len(CardBillVariable)):
    Fig2_sub = Fig2.add_subplot(3, 2, i + 1)
    Q1 = dataframe[CardBillVariable[i]].quantile(0.25)
    Q3 = dataframe[CardBillVariable[i]].quantile(0.75)
    IQR = Q3 - Q1
    sns.boxplot(x=dataframe['SEX'], y=dataframe.loc[(dataframe[CardBillVariable[i]] > Q1 - 0 * IQR)
    & (dataframe[CardBillVariable[i]] < Q3 + 0 * IQR)][CardBillVariable[i]], data=dataframe)
    pass
plt.show()
# 성별에 따른 카드 대금 청구액 시각화
PrepaidVariable = ['PREPAID_SEP', 'PREPAID_AUG', 'PREPAID_JUL', 'PREPAID_JUN', 'PREPAID_MAY', 'PREPAID_APR']
Fig3 = plt.figure(figsize = (25, 25))
for i in range(len(PrepaidVariable)):
    Fig3_sub = Fig3.add_subplot(3, 2, i + 1)
    Q1 = dataframe[PrepaidVariable[i]].quantile(0.25)
    Q3 = dataframe[PrepaidVariable[i]].quantile(0.75)
    IQR = Q3 - Q1
    sns.boxplot(x=dataframe['SEX'], y=dataframe.loc[(dataframe[PrepaidVariable[i]] > Q1 - 0.5 * IQR)
    & (dataframe[PrepaidVariable[i]] < Q3 + 0.5 * IQR)][PrepaidVariable[i]], data=dataframe)
    pass
plt.show()
# 성별에 따른 선불 결제액 시각화
Fig4 = sns.countplot(x = dataframe['SEX'], hue = dataframe['DEFAULT'], hue_order = ['Yes', 'No'], data = dataframe)
# 성별에 따른 채무불이행 여부 시각화

