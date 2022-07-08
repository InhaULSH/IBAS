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
# 상환 현황 값이 범례에서 벗어난 고객은 관측값을 수정
MinersBillCustomer = dataframe.loc[(dataframe['CARD BILL_SEP'] < 0) | (dataframe['CARD BILL_AUG'] < 0) | (dataframe['CARD BILL_JUL'] < 0) |
                                   (dataframe['CARD BILL_JUN'] < 0) | (dataframe['CARD BILL_MAY'] < 0) | (dataframe['CARD BILL_APR'] < 0)].index
dataframe.drop(MinersBillCustomer, inplace = True)
# 신용카드 청구액이 음수인 고객 삭제


sns.countplot(x = dataframe['SEX'])
plt.show()
print(dataframe['SEX'].value_counts())
# 표본에서의 성비 시각화
sns.countplot(x = dataframe['EDUCATION'], order = ['Graduate', 'Undergraduate', 'High School', 'Less than Middle School'])
plt.show()
print(dataframe['EDUCATION'].value_counts())
# 표본에서의 학력 시각화
sns.countplot(x = dataframe['MARRIAGE'])
plt.show()
print(dataframe['MARRIAGE'].value_counts())
# 표본에서의 결혼여부 시각화
dataframe.loc[dataframe[(dataframe['AGE'] >= 20) & (dataframe['AGE'] < 30)].index, 'AGE GROUP'] = '20 - 30'
dataframe.loc[dataframe[(dataframe['AGE'] >= 30) & (dataframe['AGE'] < 40)].index, 'AGE GROUP'] = '30 - 40'
dataframe.loc[dataframe[(dataframe['AGE'] >= 40) & (dataframe['AGE'] < 50)].index, 'AGE GROUP'] = '40 - 50'
dataframe.loc[dataframe[(dataframe['AGE'] >= 50) & (dataframe['AGE'] < 60)].index, 'AGE GROUP'] = '50 - 60'
dataframe.loc[dataframe[dataframe['AGE'] >= 60].index, 'AGE GROUP'] = '60+'
sns.countplot(x = dataframe['AGE GROUP'], order = ['20 - 30', '30 - 40', '40 - 50', '50 - 60', '60+'])
print(dataframe['AGE GROUP'].value_counts())
plt.show()
# 표본에서의 연령대 분포 시각화

Q1_cl = dataframe['CREDIT LIMIT'].quantile(0.25)
Q3_cl = dataframe['CREDIT LIMIT'].quantile(0.75)
IQR_cl = Q3_cl - Q1_cl
dataframe_CL = dataframe.loc[(dataframe['CREDIT LIMIT'] > Q1_cl - 0.5 * IQR_cl) & (dataframe['CREDIT LIMIT'] < Q3_cl + 0.5 * IQR_cl)]
# 신용한도 변수에서 이상치 제거
Fig0 = sns.boxplot(x = dataframe_CL['SEX'], y = dataframe_CL['CREDIT LIMIT'], data = dataframe_CL)
print(dataframe.groupby(['SEX'])['CREDIT LIMIT'].median())
# 성별에 따른 신용한도 시각화
Fig1 = plt.figure(figsize = (25, 25))
for i in range(len(RepayStatVariable)):
    Fig1_sub = Fig1.add_subplot(3, 2, i + 1)
    sns.countplot(x = dataframe['SEX'], hue = dataframe[RepayStatVariable[i]], hue_order =
    ['No Overdue', 'Short-term Overdue', '2 Months ~ 6 Months Overdue', 'Long-term Overdue'], data = dataframe)
    pass
plt.show()
# 성별에 따른 상환 현황 시각화
CardBillVariable = ['CARD BILL_SEP', 'CARD BILL_AUG', 'CARD BILL_JUL', 'CARD BILL_JUN', 'CARD BILL_MAY', 'CARD BILL_APR']
Fig2 = plt.figure(figsize = (25, 25))
for i in range(len(CardBillVariable)):
    Q1_cb = dataframe[CardBillVariable[i]].quantile(0.25)
    Q3_cb = dataframe[CardBillVariable[i]].quantile(0.75)
    IQR_cb = Q3_cb - Q1_cb
    dataframe_CB = dataframe.loc[(dataframe[CardBillVariable[i]] > Q1_cb - 0.5 * IQR_cb)
                                 & (dataframe[CardBillVariable[i]] < Q3_cb + 0.5 * IQR_cb)]
    # 신용카드 청구액 변수에서 이상치 제거
    Fig2_sub = Fig2.add_subplot(3, 2, i + 1)
    sns.boxplot(x = dataframe_CB['SEX'], y = dataframe_CB[CardBillVariable[i]], data = dataframe_CB)
    print(dataframe.groupby(['SEX'])[CardBillVariable[i]].median())
    pass
plt.show()
# 성별에 따른 신용카드 청구액 시각화
PrepaidVariable = ['PREPAID_SEP', 'PREPAID_AUG', 'PREPAID_JUL', 'PREPAID_JUN', 'PREPAID_MAY', 'PREPAID_APR']
Fig3 = plt.figure(figsize = (25, 25))
for i in range(len(PrepaidVariable)):
    Q1_pr = dataframe[PrepaidVariable[i]].quantile(0.25)
    Q3_pr = dataframe[PrepaidVariable[i]].quantile(0.75)
    IQR_pr = Q3_pr - Q1_pr
    dataframe_PR = dataframe.loc[(dataframe[PrepaidVariable[i]] > Q1_pr - 0.5 * IQR_pr)
                                 & (dataframe[PrepaidVariable[i]] < Q3_pr + 0.5 * IQR_pr)]
    # 선불결제 이용액 변수에서 이상치 제거
    Fig3_sub = Fig3.add_subplot(3, 2, i + 1)
    sns.boxplot(x = dataframe_PR['SEX'], y = dataframe_PR[PrepaidVariable[i]], data = dataframe_PR)
    print(dataframe.groupby(['SEX'])[PrepaidVariable[i]].median())
    pass
plt.show()
# 성별에 따른 선불결제 이용액 시각화
Fig4 = sns.countplot(x = dataframe['SEX'], hue = dataframe['DEFAULT'], hue_order = ['Yes', 'No'], data = dataframe)
for p in Fig4.patches:
    height = p.get_height()
    Fig4.text(p.get_x() + p.get_width() / 2., height + 3, height, ha = 'center', size = 9)
plt.show()
# 성별에 따른 채무불이행 여부 시각화

Q1_cl = dataframe['CREDIT LIMIT'].quantile(0.25)
Q3_cl = dataframe['CREDIT LIMIT'].quantile(0.75)
IQR_cl = Q3_cl - Q1_cl
dataframe_CL = dataframe.loc[(dataframe['CREDIT LIMIT'] > Q1_cl - 0.5 * IQR_cl) & (dataframe['CREDIT LIMIT'] < Q3_cl + 0.5 * IQR_cl)]
# 신용한도 변수에서 이상치 제거
Fig0 = sns.boxplot(x = dataframe_CL['EDUCATION'], y = dataframe_CL['CREDIT LIMIT'],
       order = ['Graduate', 'Undergraduate', 'High School', 'Less than Middle School'], data = dataframe_CL)
print(dataframe.groupby(['EDUCATION'])['CREDIT LIMIT'].median())
# 학력에 따른 신용한도 시각화
Fig1 = plt.figure(figsize = (25, 25))
for i in range(len(RepayStatVariable)):
    Fig1_sub = Fig1.add_subplot(3, 2, i + 1)
    sns.countplot(x = dataframe['EDUCATION'], order = ['Graduate', 'Undergraduate', 'High School', 'Less than Middle School'],
    hue = dataframe[RepayStatVariable[i]], hue_order = ['No Overdue', 'Short-term Overdue', '2 Months ~ 6 Months Overdue', 'Long-term Overdue'], data = dataframe)
    pass
plt.show()
# 학력에 따른 상환 현황 시각화
CardBillVariable = ['CARD BILL_SEP', 'CARD BILL_AUG', 'CARD BILL_JUL', 'CARD BILL_JUN', 'CARD BILL_MAY', 'CARD BILL_APR']
Fig2 = plt.figure(figsize = (25, 25))
for i in range(len(CardBillVariable)):
    Q1_cb = dataframe[CardBillVariable[i]].quantile(0.25)
    Q3_cb = dataframe[CardBillVariable[i]].quantile(0.75)
    IQR_cb = Q3_cb - Q1_cb
    dataframe_CB = dataframe.loc[(dataframe[CardBillVariable[i]] > Q1_cb - 0 * IQR_cb)
                                 & (dataframe[CardBillVariable[i]] < Q3_cb + 0 * IQR_cb)]
    # 신용카드 청구액 변수에서 이상치 제거
    Fig2_sub = Fig2.add_subplot(3, 2, i + 1)
    sns.boxplot(x = dataframe_CB['EDUCATION'], y = dataframe_CB[CardBillVariable[i]],
    order = ['Graduate', 'Undergraduate', 'High School', 'Less than Middle School'], data = dataframe_CB)
    print(dataframe.groupby(['EDUCATION'])[CardBillVariable[i]].median())
    pass
plt.show()
# 학력에 따른 신용카드 청구액 시각화
PrepaidVariable = ['PREPAID_SEP', 'PREPAID_AUG', 'PREPAID_JUL', 'PREPAID_JUN', 'PREPAID_MAY', 'PREPAID_APR']
Fig3 = plt.figure(figsize = (25, 25))
for i in range(len(PrepaidVariable)):
    Q1_pr = dataframe[PrepaidVariable[i]].quantile(0.25)
    Q3_pr = dataframe[PrepaidVariable[i]].quantile(0.75)
    IQR_pr = Q3_pr - Q1_pr
    dataframe_PR = dataframe.loc[(dataframe[PrepaidVariable[i]] > Q1_pr - 0.125 * IQR_pr)
                                 & (dataframe[PrepaidVariable[i]] < Q3_pr + 0.125 * IQR_pr)]
    # 선불결제 이용액 변수에서 이상치 제거
    Fig3_sub = Fig3.add_subplot(3, 2, i + 1)
    sns.boxplot(x = dataframe_PR['EDUCATION'], y = dataframe_PR[PrepaidVariable[i]],
    order = ['Graduate', 'Undergraduate', 'High School', 'Less than Middle School'], data = dataframe_PR)
    print(dataframe.groupby(['EDUCATION'])[PrepaidVariable[i]].median())
    pass
plt.show()
# 학력에 따른 선불결제 이용액 시각화
Fig4 = sns.countplot(x = dataframe['EDUCATION'], order = ['Graduate', 'Undergraduate', 'High School', 'Less than Middle School'],
                     hue = dataframe['DEFAULT'], hue_order = ['Yes', 'No'], data = dataframe)
for p in Fig4.patches:
    height = p.get_height()
    Fig4.text(p.get_x() + p.get_width() / 2., height + 3, height, ha = 'center', size = 9)
plt.show()
# 학력에 따른 채무불이행 여부 시각화

Q1_cl = dataframe['CREDIT LIMIT'].quantile(0.25)
Q3_cl = dataframe['CREDIT LIMIT'].quantile(0.75)
IQR_cl = Q3_cl - Q1_cl
dataframe_CL = dataframe.loc[(dataframe['CREDIT LIMIT'] > Q1_cl - 0.75 * IQR_cl) & (dataframe['CREDIT LIMIT'] < Q3_cl + 0.75 * IQR_cl)]
# 신용한도 변수에서 이상치 제거
Fig0 = sns.boxplot(x = dataframe_CL['MARRIAGE'], y = dataframe_CL['CREDIT LIMIT'], data = dataframe_CL)
print(dataframe.groupby(['MARRIAGE'])['CREDIT LIMIT'].median())
# 결혼여부에 따른 신용한도 시각화
Fig1 = plt.figure(figsize = (25, 25))
for i in range(len(RepayStatVariable)):
    Fig1_sub = Fig1.add_subplot(3, 2, i + 1)
    sns.countplot(x = dataframe['MARRIAGE'], hue = dataframe[RepayStatVariable[i]], hue_order =
    ['No Overdue', 'Short-term Overdue', '2 Months ~ 6 Months Overdue', 'Long-term Overdue'], data = dataframe)
    pass
plt.show()
# 결혼여부에 따른 상환 현황 시각화
CardBillVariable = ['CARD BILL_SEP', 'CARD BILL_AUG', 'CARD BILL_JUL', 'CARD BILL_JUN', 'CARD BILL_MAY', 'CARD BILL_APR']
Fig2 = plt.figure(figsize = (25, 25))
for i in range(len(CardBillVariable)):
    Q1_cb = dataframe[CardBillVariable[i]].quantile(0.25)
    Q3_cb = dataframe[CardBillVariable[i]].quantile(0.75)
    IQR_cb = Q3_cb - Q1_cb
    dataframe_CB = dataframe.loc[(dataframe[CardBillVariable[i]] > Q1_cb - 0.25 * IQR_cb)
                                 & (dataframe[CardBillVariable[i]] < Q3_cb + 0.25 * IQR_cb)]
    # 신용카드 청구액 변수에서 이상치 제거
    Fig2_sub = Fig2.add_subplot(3, 2, i + 1)
    sns.boxplot(x = dataframe_CB['MARRIAGE'], y = dataframe_CB[CardBillVariable[i]], data = dataframe_CB)
    print(dataframe.groupby(['MARRIAGE'])[CardBillVariable[i]].median())
    pass
plt.show()
# 결혼여부에 따른 신용카드 청구액 시각화
PrepaidVariable = ['PREPAID_SEP', 'PREPAID_AUG', 'PREPAID_JUL', 'PREPAID_JUN', 'PREPAID_MAY', 'PREPAID_APR']
Fig3 = plt.figure(figsize = (25, 25))
for i in range(len(PrepaidVariable)):
    Q1_pr = dataframe[PrepaidVariable[i]].quantile(0.25)
    Q3_pr = dataframe[PrepaidVariable[i]].quantile(0.75)
    IQR_pr = Q3_pr - Q1_pr
    dataframe_PR = dataframe.loc[(dataframe[PrepaidVariable[i]] > Q1_pr - 0.25 * IQR_pr)
                                 & (dataframe[PrepaidVariable[i]] < Q3_pr + 0.25 * IQR_pr)]
    # 선불결제 이용액 변수에서 이상치 제거
    Fig3_sub = Fig3.add_subplot(3, 2, i + 1)
    sns.boxplot(x = dataframe_PR['MARRIAGE'], y = dataframe_PR[PrepaidVariable[i]], data = dataframe_PR)
    print(dataframe.groupby(['MARRIAGE'])[PrepaidVariable[i]].median())
    pass
plt.show()
# 결혼여부에 따른 선불결제 이용액 시각화
Fig4 = sns.countplot(x = dataframe['MARRIAGE'], hue = dataframe['DEFAULT'], hue_order = ['Yes', 'No'], data = dataframe)
for p in Fig4.patches:
    height = p.get_height()
    Fig4.text(p.get_x() + p.get_width() / 2., height + 3, height, ha = 'center', size = 9)
plt.show()
# 결혼여부에 따른 채무불이행 여부 시각화

Q1_cl = dataframe['CREDIT LIMIT'].quantile(0.25)
Q3_cl = dataframe['CREDIT LIMIT'].quantile(0.75)
IQR_cl = Q3_cl - Q1_cl
dataframe_CL = dataframe.loc[(dataframe['CREDIT LIMIT'] > Q1_cl - 0.75 * IQR_cl) & (dataframe['CREDIT LIMIT'] < Q3_cl + 0.75 * IQR_cl)]
# 신용한도 변수에서 이상치 제거
Fig0 = sns.boxplot(x = dataframe_CL['AGE GROUP'], y = dataframe_CL['CREDIT LIMIT'],
        order = ['20 - 30', '30 - 40', '40 - 50', '50 - 60', '60+'], data = dataframe_CL)
print(dataframe.groupby(['AGE GROUP'])['CREDIT LIMIT'].median())
# 연령대에 따른 신용한도 시각화
Fig1 = plt.figure(figsize = (25, 25))
for i in range(len(RepayStatVariable)):
    Fig1_sub = Fig1.add_subplot(3, 2, i + 1)
    sns.countplot(x = dataframe['AGE GROUP'], order = ['20 - 30', '30 - 40', '40 - 50', '50 - 60', '60+'],
    hue = dataframe[RepayStatVariable[i]], hue_order = ['No Overdue', 'Short-term Overdue', '2 Months ~ 6 Months Overdue', 'Long-term Overdue'], data = dataframe)
    pass
plt.show()
# 연령대에 따른 상환 현황 시각화
CardBillVariable = ['CARD BILL_SEP', 'CARD BILL_AUG', 'CARD BILL_JUL', 'CARD BILL_JUN', 'CARD BILL_MAY', 'CARD BILL_APR']
Fig2 = plt.figure(figsize = (25, 25))
for i in range(len(CardBillVariable)):
    Q1_cb = dataframe[CardBillVariable[i]].quantile(0.25)
    Q3_cb = dataframe[CardBillVariable[i]].quantile(0.75)
    IQR_cb = Q3_cb - Q1_cb
    dataframe_CB = dataframe.loc[(dataframe[CardBillVariable[i]] > Q1_cb - 0.25 * IQR_cb)
                                 & (dataframe[CardBillVariable[i]] < Q3_cb + 0.25 * IQR_cb)]
    # 신용카드 청구액 변수에서 이상치 제거
    Fig2_sub = Fig2.add_subplot(3, 2, i + 1)
    sns.boxplot(x = dataframe_CB['AGE GROUP'], y = dataframe_CB[CardBillVariable[i]],
    order = ['20 - 30', '30 - 40', '40 - 50', '50 - 60', '60+'], data = dataframe_CB)
    print(dataframe.groupby(['AGE GROUP'])[CardBillVariable[i]].median())
    pass
plt.show()
# 연령대에 따른 신용카드 청구액 시각화
PrepaidVariable = ['PREPAID_SEP', 'PREPAID_AUG', 'PREPAID_JUL', 'PREPAID_JUN', 'PREPAID_MAY', 'PREPAID_APR']
Fig3 = plt.figure(figsize = (25, 25))
for i in range(len(PrepaidVariable)):
    Q1_pr = dataframe[PrepaidVariable[i]].quantile(0.25)
    Q3_pr = dataframe[PrepaidVariable[i]].quantile(0.75)
    IQR_pr = Q3_pr - Q1_pr
    dataframe_PR = dataframe.loc[(dataframe[PrepaidVariable[i]] > Q1_pr - 0.25 * IQR_pr)
                                 & (dataframe[PrepaidVariable[i]] < Q3_pr + 0.25 * IQR_pr)]
    # 선불결제 이용액 변수에서 이상치 제거
    Fig3_sub = Fig3.add_subplot(3, 2, i + 1)
    sns.boxplot(x = dataframe_PR['AGE GROUP'], y = dataframe_PR[PrepaidVariable[i]],
    order = ['20 - 30', '30 - 40', '40 - 50', '50 - 60', '60+'], data = dataframe_PR)
    print(dataframe_PR.groupby(['AGE GROUP'])[PrepaidVariable[i]].median())
    pass
plt.show()
# 연령대에 따른 선불결제 이용액 시각화
Fig4 = sns.countplot(x = dataframe['AGE GROUP'], order = ['20 - 30', '30 - 40', '40 - 50', '50 - 60', '60+'],
                     hue = dataframe['DEFAULT'], hue_order = ['Yes', 'No'], data = dataframe)
for p in Fig4.patches:
    height = p.get_height()
    Fig4.text(p.get_x() + p.get_width() / 2., height + 3, height, ha = 'center', size = 9)
plt.show()
# 연령대에 따른 채무불이행 여부 시각화