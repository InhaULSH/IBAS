# 5. Scikit-Learn 기본
import sklearn
import numpy as np
import pandas as pd
import xgboost
import lightgbm
from sympy.abc import lamda
# Feature = 타겟값을 제외한 나머지 속성
# Target Value = 지도학습의 정답 데이터
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

Iris = load_iris()
# 유사 딕셔너리 { Feature : [ndarray], Target Value : [ndarray] } 형태의 데이터셋 반환
# data : 피쳐 데이터셋, target : 타겟값 데이터셋, target_name : 타겟값의 속성명, feature_names : 피쳐 속성명, DESCR : 데이터셋 및 피쳐 설명
Iris_Features = Iris.data
Iris_Target = Iris.target
X_train, X_test, Y_train, Y_test = train_test_split(Iris_Features, Iris_Target, test_size=0.2, random_state=11)
# 학습 데이터와 테스트 데이터 분리, random_state는 결과값 고정을 위한 것, 없으면 매 시행마다 무작위 표본 추출
Iris_df = pd.DataFrame(data=Iris.data, columns=Iris.feature_names)
Iris_df['Target'] = Iris.target
Feature_df = Iris_df.iloc[:, :-1]
Target_df = Iris_df.iloc[:, -1]
# X_train, X_test, Y_train, Y_test = train_test_split(Feature_df, Target_df, test_size=0.2, random_state=11)
# Data Frame 기반으로도 데이터셋 선언 및 분리 가능

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
# 예측 정확도 평가 전 학습 데이터의 부분집합인 검증 데이터로 여러번 검증
kfold = KFold(n_splits=5, shuffle=True)
dt_clf = DecisionTreeClassifier()
cv_accuracy = []
n_iter = 0
for train_index, test_index in kfold.split(Iris_Features):
    # kfold.split( )으로 반환된 인덱스를 이용하여 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = Iris_Features[train_index], Iris_Features[test_index]
    Y_train, Y_test = Iris_Target[train_index], Iris_Target[test_index]

    #학습 및 예측
    dt_clf.fit(X_train , Y_train)
    pred = dt_clf.predict(X_test)
    n_iter += 1

    # 반복 시 마다 정확도 측정
    accuracy = np.round(accuracy_score(Y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'
          .format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter,test_index))

    cv_accuracy.append(accuracy)
# 일반 K-Fold 교차 검즘 : K는 학습데이터를 K개의 검증데이터로 분리하여 K회 반복 검증
from sklearn.model_selection import StratifiedKFold
skfold = StratifiedKFold(n_splits=3)
n_iter = 0
cv_accuracy = []
for train_index, test_index in skfold.split(Iris_Features, Iris_Target):
    # split( )으로 반환된 인덱스를 이용하여 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = Iris_Features[train_index], Iris_Features[test_index]
    Y_train, Y_test = Iris_Target[train_index], Iris_Target[test_index]

    # 학습 및 예측
    dt_clf.fit(X_train, Y_train)
    pred = dt_clf.predict(X_test)
    n_iter += 1

    # 반복 시 마다 정확도 측정
    accuracy = np.round(accuracy_score(Y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'
          .format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter, test_index))
    cv_accuracy.append(accuracy)
# Straitified K-Fold 교차 검증 : 분류의 경우 학습데이터의 분포가 불균등할 때 학습데이터와 분포가 유사하도록 검증데이터 추출

df_Clf = DecisionTreeClassifier(random_state=11)
# 결정 트리 분류 Estimator 생성
df_Clf.fit(X_train, Y_train)
# 생성한 Estimator로 학습 데이터로 학습 수행
Pred = df_Clf.predict(X_test)
print('예측 정확도: {0:.4f}'.format(accuracy_score(Y_test,Pred)))
# 테스트 데이터로 예측 후 예측 정확도 평가
# 학습 데이터로 예측하는 것은 의미가 없음, 테스트 데이터로 예측해야
from sklearn.model_selection import cross_val_score
Scores = cross_val_score(dt_clf , Iris_Features , Iris_Target , scoring = 'accuracy', cv = 3)
print('교차 검증별 정확도:',np.round(Scores, 4))
print('평균 검증 정확도:', np.round(np.mean(Scores), 4))
# cross_val_score() : Straitified K-Fold 검증 절차(폴드 세트 추출, 학습, 예측, 검증)을 한 함수로 수행 가능
from sklearn.model_selection import GridSearchCV, train_test_split
HyperParameters = {'max_depth':[1, 2, 3], 'min_samples_split':[2,3]}
dtree = DecisionTreeClassifier()
grid_dtree = GridSearchCV(dtree, param_grid = HyperParameters, cv = 3, refit = True, return_train_score = True)
# GridSearchCV 함수는 최적 파라미터로 학습된 결과값을 자동 반환하므로 결과값 학습/예측/검증/평가 가능
from sklearn.metrics import accuracy_score
grid_dtree.fit(X_train, Y_train)
Pred_GSCV = grid_dtree.predict(X_test)
print('테스트 데이터 세트 정확도: {0:.4f}'.format(accuracy_score(Y_test, Pred_GSCV)))
# GridSearchCV : 분류나 회귀 등 지도학습의 경우 하이퍼파리미터를 조정해가며 최적값을 찾아 알고리즘을 최적화하는데
# 각 케이스 별 교차검증 및 최적 파라미터 추출을 한 함수로 수행가능

# 데이터 인코딩
# 머신러닝 입력은 오직 결측치가 아닌숫자만 가능
from sklearn.preprocessing import LabelEncoder
Items = ['TV','냉장고','전자렌지','컴퓨터','선풍기','선풍기','믹서','믹서']
Encoder = LabelEncoder()
Encoder.fit(Items)
Labels = Encoder.transform(Items)
# 범주형 -> 정수형 : Label Encoding 필요
Items = np.array(Items).reshape(-1, 1)
from sklearn.preprocessing import OneHotEncoder
OH_Encoder = OneHotEncoder()
OH_Encoder.fit(Items)
OH_Labels = OH_Encoder.transform(Items)
# OneHotEncoder로 변환한 결과는 Sparse Matrix이므로 Dense Matrix로 변환.
OH_Labels = OH_Labels.toarray()

df_Items = pd.DataFrame({'Item':['TV','냉장고','전자렌지','컴퓨터','선풍기','선풍기','믹서','믹서'] })
dummy_Items = pd.get_dummies(df_Items)
# Incodig된 값은 서열이 있거나 연산이 가능하지 않으므로 원핫 인코딩 필요(= 더미변수 이용)

# 데이터 스케일링
from sklearn.preprocessing import StandardScaler
Iris_df_NotScaled = pd.DataFrame(data=Iris.data, columns=Iris.feature_names)
Scaler = StandardScaler()
Scaler.fit(Iris_df_NotScaled)
Iris_Scaled = Scaler.transform(Iris_df_NotScaled)
Iris_df_Scaled = pd.DataFrame(data=Iris_Scaled, columns=Iris.feature_names)
print(Iris_df_Scaled.mean())
print(Iris_df_Scaled.var())
# 피쳐 표준화 : 평균 0, 분산 1인 정규분포상로 변환 X' = X - Mean / STD
from sklearn.preprocessing import MinMaxScaler
Iris_df_NotMmScaled = pd.DataFrame(data=Iris.data, columns=Iris.feature_names)
scaler = MinMaxScaler()
scaler.fit(Iris_df_NotMmScaled)
Iris_Mmscaled = scaler.transform(Iris_df_NotMmScaled)
Iris_df_MmScaled = pd.DataFrame(data=Iris_Mmscaled, columns=Iris.feature_names)
print(Iris_df_MmScaled.min())
print(Iris_df_MmScaled.max())
# 피쳐 정규화 : 서로 다른 피쳐 크기를 -1에서 1사이로 변환 X' = X - Min / Max - Min
train_array = np.arange(0, 11).reshape(-1, 1)
test_array =  np.arange(0, 6).reshape(-1, 1)
scaler = MinMaxScaler()
scaler.fit(train_array)
train_scaled = scaler.transform(train_array)
print('원본 train_array 데이터:', np.round(train_array.reshape(-1), 2))
print('Scale된 train_array 데이터:', np.round(train_scaled.reshape(-1), 2))
# test_array에 Scale 변환을 할 때는 반드시 fit()을 호출하지 않고 transform() 만으로 변환해야 함.
test_scaled = scaler.transform(test_array)
print('\n원본 test_array 데이터:', np.round(test_array.reshape(-1), 2))
print('Scale된 test_array 데이터:', np.round(test_scaled.reshape(-1), 2))
# 정규화 시 학습데이터의 척도와 테스트 데이터의 척도는 동일해야 함, 같은 값이면 정규화된 값도 같아함

Titanic = pd.read_csv('./DataSet_MachineLearning/Visual/titanic_train.csv')
# Target Value는 survived(생존여부)
print(Titanic.info())
print(Titanic.describe().transpose())
# Age 결측치는 평균값으로, 나머지 결측치는 NULL 문자열로
Titanic['Age'] = Titanic['Age'].fillna(Titanic['Age'].mean())
Titanic['Cabin'] = Titanic['Cabin'].fillna('NULL')
Titanic['Embarked'] = Titanic['Embarked'].fillna('NULL')
# Cabin의 등급만 나타내도록 대체
# df 수정시에는 df[] = df[].함수() 형태로 명시적으로 수정해야함 아닐 경우 오류발생 가능성 있음
Titanic['Cabin'] = Titanic['Cabin'].str[:1]
# 관측하지 않을 Feature는 Drop
Titanic = Titanic.drop(['PassengerId','Name','Ticket'], axis=1)
# 관측할 Feature 3개를 정수형으로 Encoding
from sklearn.preprocessing import LabelEncoder
def encode_features(df):
    features = ['Cabin', 'Sex', 'Embarked']
    le = LabelEncoder()
    for feature in features:
        le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df
Titanic = encode_features(Titanic)
# 데이터 재로딩 및 피쳐, 타겟값 추출
Titanic_df = Titanic
Y_Titanic_df = Titanic_df['Survived']
X_Titanic_df = Titanic_df.drop('Survived',axis=1, inplace=False)
X_train, X_test, Y_train, Y_test = train_test_split(X_Titanic_df, Y_Titanic_df, test_size=0.2, random_state=11)
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train , Y_train)
dt_pred = dt_clf.predict(X_test)
print('DecisionTreeClassifier 정확도: {0:.4f}'.format(accuracy_score(Y_test, dt_pred)))
# 학습/예측/평가
Titanic_scores = cross_val_score(dt_clf, X_Titanic_df , Y_Titanic_df , cv=5)
for iter_count,accuracy in enumerate(Titanic_scores):
    print("교차 검증 {0} 정확도: {1:.4f}".format(iter_count, accuracy))
print("평균 정확도: {0:.4f}".format(np.mean(Titanic_scores)))
# 교차 검증
from sklearn.model_selection import GridSearchCV
parameters = {'max_depth':[2,3,5,10],
             'min_samples_split':[2,3,5], 'min_samples_leaf':[1,5,8]}
grid_dclf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv=5)
grid_dclf.fit(X_train, Y_train)
print('GridSearchCV 최적 하이퍼 파라미터 :', grid_dclf.best_params_)
print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_dclf.best_score_))
best_dclf = grid_dclf.best_estimator_
dpredictions = best_dclf.predict(X_test)
accuracy = accuracy_score(Y_test , dpredictions)
print('테스트 세트에서의 DecisionTreeClassifier 정확도 : {0:.4f}'.format(accuracy))
# GridSearchCV 실행

# 6. 평가

# 정확도 = 예측 결과와 동일한 데이터 / 전체 예측 데이터
# 불균형한 분포를 가진 레이블 값 예측에는 부적합, 정확도만 활용하는 경우 거의 없음
from sklearn.metrics import accuracy_score
print('정확도는: {0:.4f}'.format(accuracy_score(Y_test , dpredictions)))

# 오차행렬, Confusion Matrix = 진양성, 진음성, 위양성, 위음성의 비율을 2 x 2 행렬로 파악 가능
# 사분면 상에서의 값을 통해 모델의 문제점 파악 가능
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test , dpredictions)

# 정확도 = TP / (FP + TP), 예측을 Positive로 한 케이스 중 실제로도 Positive인 케이스의
# 정확도는 위양성, FP 발생 시 업무상 큰 영향이 발생하는 경우에 더 적합한 평가지표 예를들어 스팸메일 탐지 등
# 재현율 = TP / (FN + TP), 실제로 Positive한 케이스 중 예측값이 Positive인 케이스의 비율
# 재현율은 위음성, FN 발생 시 업무상 큰 영향이 발생하는 경우에 더 적합한 평가지표 예를들어 암 진단이나 금융사기 탐지 등
# 분류 모델의 성능평가에 자주 사용됨
from sklearn.metrics import accuracy_score, precision_score , recall_score
print("정밀도:", precision_score(Y_test , dpredictions))
print("재현율:", recall_score(Y_test , dpredictions))
# 분류 모델에서는 확률을 기준으로 판단, 특정 임계값보다 작으면 무조건 음성으로 반대의 경우 무조건 양성으로 분류
# 이 값을 결정임계값이라고함, 결정임계값 감소하면 Positive로 결정할 확률이 더 높아지므로 재현율은 증가
# 정밀도 및 재현율은 트레이드오프 관계이므로 한 수치가 증가하면 다른 수치는 감소함, 상호보완적 관계

# 정밀도은 확실한 케이스만 Positive로 판단하고 나머지를 Negative로 판단하면 100%로 만들 수 있음
# 재현율은 모든 케이스를 Positive로 판단하면 100%로 만들 수 있음, 지표가 극단적으로 치우치면 오히려 문제가 있는 것
# F1 Score = 2 * (정밀도 * 재현율) / (정밀도 + 재현율), 정밀도와 재현율의 균형이 맞을수록 더 높음
from sklearn.metrics import f1_score
f1 = f1_score(Y_test , dpredictions)
print('F1 스코어: {0:.4f}'.format(f1))
# ROC 곡선은 임계값이 변화함에 따른 FP율 = FP / (FP + TN)의 변동에 대한 TP율 = 재현율의 변동을 곡선으로 나타낸 것
# AUC = ROC 곡선 아래의 면적을 계산한 것으로서 1에 가까울 수록 좋은 수치
# 곡선의 기울기가 완만할수록 AUC가 개선되므로 FP율 감소에 따라 TP율이 상대적으로 완만하게 감소할수록 좋음
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, Y_train)
pred_proba = dt_clf.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(Y_test, pred_proba)
print('ROC AUC 값: {0:.4f}'.format(roc_score))

Diabetes = pd.read_csv('./DataSet_MachineLearning/Visual/diabetes.csv')
X = Diabetes.iloc[:, :-1]
Y = Diabetes.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=156, stratify=Y)
lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, Y_train)
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:, 1]
# 데이터 불러오기 및 학습, 예측
def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix( y_test, pred)
    accu = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    f1 = f1_score(y_test,pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
    F1: {3:.4f}, AUC:{4:.4f}'.format(accu, precision, recall, f1, roc_auc))
get_clf_eval(Y_test , pred, pred_proba)
# 성능 평가 지표 계산
zero_features = ['Glucose', 'BloodPressure','SkinThickness','Insulin','BMI']
total_count = Diabetes['Glucose'].count()
for feature in zero_features:
    zero_count = Diabetes[Diabetes[feature] == 0][feature].count()
    print('{0} 0 건수는 {1}, 퍼센트는 {2:.2f} %'.format(feature, zero_count, 100*zero_count/total_count))
Diabetes[zero_features] = Diabetes[zero_features].replace(0, Diabetes[zero_features].mean())
# 값이 0인 이상치 확인 및 처리
x = Diabetes.iloc[:, :-1]
y = Diabetes.iloc[:, -1]
scaler = StandardScaler( )
# 피쳐 데이터셋에 일괄 스케일링
X_scaled = scaler.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 156, stratify=y)
lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train , y_train)
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:, 1]
get_clf_eval(y_test , pred, pred_proba)
# 0을 평균값으로 대체 후 스케일링 한 데이터셋의 성능 평가 지표 계산