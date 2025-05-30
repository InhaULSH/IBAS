# 8. 지도학습 - 회귀
# 여러 독립변수와 한 종속변수간의 상관관계, Y = W1 * X1 + W2 * X2....., 주어진 피쳐와 결정값으로 최적 회귀계수 찾기
# 잔차의 제곱합 즉 비용함수가 최소가 되는 회귀계수 추정 = 최소제곱법, 지도학습이므로 중심변수는 W0, W1... 등

# 경사하강법 = 비용함수의 반환값 즉 잔차가 작아지는 방향으로 값을 업데이트하되 더이상 작아지지 않으면 최적값으로 간주
# 미분된 기울기로 비용함수의 증감을 파악, 이때 비용함수를 W0, W1로 편미분 = 잔차합과 관련된 식 = 0
# 편미분된 수식에 의해 회귀식은 표본평균 지남, W1 = 두 변수의 공분산 / 독립변수 X의 분산
# 편미분된 수식에 일정한 학습률 N을 곱한만큼 W0와 W1을 각각 변화시켜가며 비용함수의 증감을 파악 => Local Maximum 문제
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
X = 2 * np.random.rand(100,1)
Y = 6 +4 * X+ np.random.randn(100,1)
def get_weight_updates(w1, w0, x, y, learning_rate=0.01):
    N = len(y)
    w1_update = np.zeros_like(w1)
    w0_update = np.zeros_like(w0)
    y_pred = np.dot(x, w1.T) + w0
    diff = y - y_pred
    w0_factors = np.ones((N, 1))
    w1_update = -(2 / N) * learning_rate * (np.dot(x.T, diff))
    w0_update = -(2 / N) * learning_rate * (np.dot(w0_factors.T, diff))
    return w1_update, w0_update
def gradient_descent_steps(x, y, iters=10000):
    w0 = np.zeros((1, 1))
    w1 = np.zeros((1, 1))
    for ind in range(iters):
        w1_update, w0_update = get_weight_updates(w1, w0, x, y, learning_rate=0.01)
        w1 = w1 - w1_update
        w0 = w0 - w0_update
    return w1, w0
def get_cost(y, y_pred):
    N = len(y)
    cost = np.sum(np.square(y - y_pred))/N
    return cost
w1, w0 = gradient_descent_steps(X, Y, iters=1000)
print("w1:{0:.3f} w0:{1:.3f}".format(w1[0,0], w0[0,0]))
Y_pred = w1[0,0] * X + w0
print('Gradient Descent Total Cost:{0:.4f}'.format(get_cost(Y, Y_pred)))
# 독립변수 X 전체를 경사하강법에 활용 시 비효율적, 시간 및 자원 소모 심함 -> 미니 배치 경사하강법 활용 가능
def stochastic_gradient_descent_steps(x, y, batch_size=10, iters=1000):
    w0 = np.zeros((1, 1))
    w1 = np.zeros((1, 1))
    prev_cost = 100000
    iter_index = 0
    for ind in range(iters):
        np.random.seed(ind)
        stochastic_random_index = np.random.permutation(X.shape[0])
        sample_x = x[stochastic_random_index[0:batch_size]]
        sample_y = y[stochastic_random_index[0:batch_size]]
        w1_update, w0_update = get_weight_updates(w1, w0, sample_x, sample_y, learning_rate=0.01)
        w1 = w1 - w1_update
        w0 = w0 - w0_update
    return w1, w0
w1, w0 = stochastic_gradient_descent_steps(X, Y, iters=1000)
Y_pred = w1[0,0] * X + w0
print('Stochastic Gradient Descent Total Cost:{0:.4f}'.format(get_cost(Y, Y_pred)))
# 독립변수 X 중 일부한 경사하강법에 사용

# LinearRegression 클래스 = fit()으로 독립변수와 종속 변수 입력받아 OLS로 추정후 추정계수를 coef_ 클래스에 저장
# 선형회귀의 가정 1. 모든 독립변수는 서로 독립적, 상관관계 낮음 => 상관관계 높은 독립변수 다수 존재 시 규제 적용 필요
# 선형회귀의 가정 2. 선형결합, 기본적으로 선형관계만 추정가능 => 변수에 비선형변환은 가능
# 선형회귀의 가정 3. 외생성, 관측하지 않은 변수의 영향은 무시
# 선형회귀의 가정 4. 등분산성, 관측하지 않은 효과의 의한 변동폭은 상수
# 선형회귀의 평가지표 1. MSE => 실제값과 예측값의 차이 제곱의 평균, MAE => 실제값과 예측값의 차이 절댓값의 평균
# RMSE => MSE^1/2 , RMSE/MSE는 MAE에 비해 더 큰 오류값에 더 큰 패널티를 부여
# 선형회귀의 평가지표 2. R^2 => Goodnes of Fit, 종속변수 전체 변동 중 예측모델에 의해 설명된 변동의 비율
# 추정계수의 유의성 평가가 아님, R^2가 낮아도 유의한 분석일수 있음
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score
boston = load_boston()
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df['PRICE'] = boston.target
Y_target = boston_df['PRICE']
X_data = boston_df.drop(['PRICE'], axis=1,inplace=False)
X_train , X_test , Y_train , Y_test = train_test_split(X_data , Y_target ,test_size=0.3, random_state=156)
lr = LinearRegression()
lr.fit(X_train ,Y_train)
y_preds = lr.predict(X_test)
mse = mean_squared_error(Y_test, y_preds)
rmse = np.sqrt(mse)
print('MSE : {0:.3f} , RMSE : {1:.3F}'.format(mse , rmse))
print('Variance score : {0:.3f}'.format(r2_score(Y_test, y_preds)))
print('절편 값:',lr.intercept_)
print('회귀 계수값:', np.round(lr.coef_, 1))
# cross_val_score, GridSearchCV 시 일관성을 위해 MAE도 음수처리하여 사용, MAE 자체가 음수가 가능한 것은 아님
from sklearn.model_selection import cross_val_score
neg_mse_scores = cross_val_score(lr, X_data, Y_target, scoring="neg_mean_squared_error", cv = 5)
rmse_scores  = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)
print(' 5 folds 의 개별 Negative MSE scores: ', np.round(neg_mse_scores, 2))
print(' 5 folds 의 개별 RMSE scores : ', np.round(rmse_scores, 2))
print(' 5 folds 의 평균 RMSE : {0:.3f} '.format(avg_rmse))

# 다항회귀 = 여러 독립변수, 하나의 종속변수 = 회귀식 Y = X1 * W1 + X2 * W2 + X3 * W3..... 와 같은 형태인 경우
# 실제 데이터는 독립변수간 상호작용이 존재하므로 Y = X1 * W1 + X2 * W2 + X1 * X2 * W3... 와 같은 형태를 띔
# 사이킷런에는 다항회귀 클래스 없음 PolynomialFeatures 클래스 동해 단항 피쳐를 다항 피쳐로 변환 후 선형회귀 클래스 적용
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
def polynomial_func(x):
    y = 1 + 2*x[:,0] + 3*x[:,0]**2 + 4*x[:,1]**3
    return y
model = Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', LinearRegression())])
X = np.arange(4).reshape(2,2)
Y = polynomial_func(X)
model = model.fit(X, Y)
print('Polynomial 회귀 계수\n', np.round(model.named_steps['linear'].coef_, 2))
X_train , X_test , Y_train , Y_test = train_test_split(X_data , Y_target ,test_size=0.3, random_state=156)
model = Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', LinearRegression())])
model.fit(X_train, Y_train)
y_preds = model.predict(X_test)
mse = mean_squared_error(Y_test, y_preds)
rmse = np.sqrt(mse)
print('MSE : {0:.3f} , RMSE : {1:.3F}'.format(mse , rmse))
print('Variance score : {0:.3f}'.format(r2_score(Y_test, y_preds)))
# 다항회귀의 과대적합, 과소적합 => 다항회귀식의 차수가 너무 낮으면 실제 함수에 비해 지나치게 단순하여 예측성능 떨어짐
# 반대로 너무 높으면 실제 함수와 괴리가 발생하여 주어진 표본에만 지나치게 최적화 => 편향이 높으면 분산이 낮아져 과소적
# 합 가능성높음, 반대로 분산이 커지면 편향이 낮아져 과대적합의 가능성 높음 => 편향/분산 트레이드 오프

# 규제 선형 회귀 => 잔차 제곱합 최소화 + 회귀계수 크기 제어 => Min (RSS(W) + alpha * ||W||) => alpha가 클 경우
# 회귀계수의 증가에 큰 패널티 부여, alpha가 작을 경우 RSS 최소화에 초점
# L2 규제 => 릿지회귀 : W의 제곱에 패널티 부여, 계수의 크기 조정 vs L1 규제 => 라쏘 회귀 : W의 절댓값에 패널티 부여,
# 피쳐의 개수 축소 vs L1 + L2 => 엘라스틱 네트 : 계수의 크기 조정과 피쳐의 개수 축소 동시에 실행
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
boston = load_boston()
bostonDF = pd.DataFrame(boston.data , columns = boston.feature_names)
bostonDF['PRICE'] = boston.target
Y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'],axis=1,inplace=False)
ridge = Ridge(alpha = 10)
neg_mse_scores = cross_val_score(ridge, X_data, Y_target, scoring="neg_mean_squared_error", cv = 5)
rmse_scores  = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)
print(' 5 folds 의 개별 Negative MSE scores: ', np.round(neg_mse_scores, 3))
print(' 5 folds 의 개별 RMSE scores : ', np.round(rmse_scores,3))
print(' 5 folds 의 평균 RMSE : {0:.3f} '.format(avg_rmse))
fig , axs = plt.subplots(figsize=(18, 6) , nrows=1 , ncols=5)
alphas = [0 , 0.1 , 1 , 10 , 100]
for pos , alpha in enumerate(alphas) :
    ridge = Ridge(alpha = alpha)
    ridge.fit(X_data , Y_target)
    coeff = pd.Series(data=ridge.coef_ , index=X_data.columns )
    colname='alpha:'+str(alpha)
    coeff_df[colname] = coeff
    coeff = coeff.sort_values(ascending=False)
    axs[pos].set_title(colname)
    axs[pos].set_xlim(-20, 6)
    sns.barplot(x=coeff.values , y=coeff.index, ax=axs[pos])
plt.show()
ridge_alphas = [0 , 0.1 , 1 , 10 , 100]
sort_column = 'alpha:'+str(ridge_alphas[0])
coeff_df.sort_values(by=sort_column, ascending=False)
# 릿지 회귀 = L2 규제 : alpha * |W|^2 으로 패널티 부여, 회귀계수의 크기를 감소 시킴
# alpha와 회귀 성능은 항상 비례하는 것은 아님
from sklearn.linear_model import Lasso, ElasticNet
def get_linear_reg_eval(model_name, params=None, X_data_n=None, y_target_n=None, verbose=True, return_coeff=True):
    coeff_df = pd.DataFrame()
    if verbose: print('####### ', model_name, '#######')
    for param in params:
        if model_name == 'Ridge':
            model = Ridge(alpha=param)
        elif model_name == 'Lasso':
            model = Lasso(alpha=param)
        elif model_name == 'ElasticNet':
            model = ElasticNet(alpha=param, l1_ratio=0.7)
        neg_mse_scores = cross_val_score(model, X_data_n, y_target_n, scoring="neg_mean_squared_error", cv=5)
        avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))
        print('alpha {0}일 때 5 폴드 세트의 평균 RMSE: {1:.3f} '.format(param, avg_rmse))
        model.fit(X_data_n, y_target_n)
        if return_coeff:
            coeff = pd.Series(data=model.coef_, index=X_data_n.columns)
            colname = 'alpha:' + str(param)
            coeff_df[colname] = coeff
    return coeff_df
lasso_alphas = [ 0.07, 0.1, 0.5, 1, 3]
coeff_lasso_df =get_linear_reg_eval('Lasso', params=lasso_alphas, X_data_n=X_data, y_target_n=Y_target)
sort_column = 'alpha:'+str(lasso_alphas[0])
coeff_lasso_df.sort_values(by=sort_column, ascending=False)
elastic_alphas = [ 0.07, 0.1, 0.5, 1, 3]
coeff_elastic_df =get_linear_reg_eval('ElasticNet', params=elastic_alphas, X_data_n=X_data, y_target_n=Y_target)
sort_column = 'alpha:'+str(elastic_alphas[0])
coeff_elastic_df.sort_values(by=sort_column, ascending=False)
# 라쏘 회귀 = L1 규제 : alpha * |W|으로 패널티 부여, 불필요한 회귀 계수의 크기 급격히 감소시켜 0으로 만듬, 피쳐 셀렉
# 션 기능 수행 / 엘라스틱 넷 회귀 = L1 + L2 : 라쏘 회귀의 alpha 값에 따른 회귀계수의 급격한 변동 현상을 보완하기 위해
# L2 규제를 라쏘 회귀에 적용, L1의 alpha를 a라하고 L2의 alpha를 b라하면 엘라스틱넷 회귀의 alpha는 a + b

# 선형회귀의 경우 피쳐값 및 타겟값이 정규분포를 이룬다고 가정 => 회귀의 용이성을 위해 데이터 변환 필요함
# 정규화 : 표준정규분포 형태(StandardScaler) 또는 0과 1사이로(MinMaxScaler)
# 다항변환 : 정규화 또는 스케일링된 데이터셋에 다항 특성 적용(PolynomialFeatures)
# 로그변환 : 왜도가 심한 데이터는 피쳐 또는 타켓 데이터에 로그 적용, 어디에 적용하는 지에 따라 결과값의 의미 달라짐
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
def get_scaled_data(method='None', p_degree=None, input_data=None):
    if method == 'Standard':
        scaled_data = StandardScaler().fit_transform(input_data)
    elif method == 'MinMax':
        scaled_data = MinMaxScaler().fit_transform(input_data)
    elif method == 'Log':
        scaled_data = np.log1p(input_data)
    else:
        scaled_data = input_data
    if p_degree != None:
        scaled_data = PolynomialFeatures(degree=p_degree, include_bias=False).fit_transform(scaled_data)
    return scaled_data
alphas = [0.1, 1, 10, 100]
scale_methods=[(None, None), ('Standard', None), ('Standard', 2), ('MinMax', None), ('MinMax', 2), ('Log', None)]
for scale_method in scale_methods:
    X_data_scaled = get_scaled_data(method=scale_method[0], p_degree=scale_method[1], input_data=X_data)
    print('\n## 변환 유형:{0}, Polynomial Degree:{1}'.format(scale_method[0], scale_method[1]))
    get_linear_reg_eval('Ridge', params=alphas, X_data_n=X_data_scaled, y_target_n=Y_target, verbose=False, return_coeff=False)

# 로지스틱 회귀 => 독립변수와 종속변수를 축으로하는 공간에 표본이 존재할때 트리로 해당 표본을 분류하지 않고 회귀직선으로
# 해당 표본을 분류, 회귀직선에서 거리가 멀 수록 종속변수가 0 또는 1일 확률이 높은 것, 이진 분류에 사용하나 이진 분류가
# 아닌 경우에도 사용 가능, 음의 무한대에서 양의 무한대의 입력이 있을때 0에서 1 사이로 출력하는 시그모이드 함수를 사용
# 시그모이드 함수의 X축은 회귀직선으로부터 해당 표본까지의 거리, Y축은 해당 표본에 대해 종속변수가 1일 확률
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
cancer = load_breast_cancer()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(cancer.data)
X_train , X_test, y_train , Y_test = train_test_split(data_scaled, cancer.target, test_size=0.3, random_state=0)
lr_clf = LogisticRegression(penalty='l1', C=0.1)
# penalty는 적용할 규제 종류, C는 alpha값의 역수
lr_clf.fit(X_train, y_train)
lr_preds = lr_clf.predict(X_test)
print('accuracy: {0:.3f}, roc_auc:{1:.3f}'.format(accuracy_score(Y_test, lr_preds), roc_auc_score(Y_test , lr_preds)))
solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
for solver in solvers:
    lr_clf = LogisticRegression(solver=solver, max_iter=600)
    lr_clf.fit(X_train, y_train)
    lr_preds = lr_clf.predict(X_test)
    print('solver:{0}, accuracy: {1:.3f}, roc_auc:{2:.3f}'.format(solver, accuracy_score(Y_test, lr_preds), roc_auc_score(Y_test , lr_preds)))
params={'solver':['liblinear', 'lbfgs'], 'penalty':['l2', 'l1'], 'C':[0.01, 0.1, 1, 1, 5, 10]}
# solver는 회귀계수 최적화 방식
grid_clf = GridSearchCV(lr_clf, param_grid=params, scoring='accuracy', cv=3 )
grid_clf.fit(data_scaled, cancer.target)
print('최적 하이퍼 파라미터:{0}, 최적 평균 정확도:{1:.3f}'.format(grid_clf.best_params_, grid_clf.best_score_))

# 회귀 트리 => 독립변수를 결정트리를 통해 여러 구간으로 분할한 후 각 구간의 평균값으로 각각 회귀 = CART 기법, 복잡한 데
# 이터를 적절하게 회귀할 수도 있으나 트리의 깊이가 너무 깊을 경우 다항회귀의 과대적합 문제가 유사하게 발생
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
boston = load_boston()
bostonDF = pd.DataFrame(boston.data, columns = boston.feature_names)
bostonDF['PRICE'] = boston.target
y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'], axis=1,inplace=False)
dt_reg = DecisionTreeRegressor(random_state=0, max_depth=4)
rf_reg = RandomForestRegressor(random_state=0, n_estimators=1000)
gb_reg = GradientBoostingRegressor(random_state=0, n_estimators=1000)
xgb_reg = XGBRegressor(n_estimators=1000)
lgb_reg = LGBMRegressor(n_estimators=1000)
def get_model_cv_prediction(model, X_data, y_target):
    neg_mse_scores = cross_val_score(model, X_data, y_target, scoring="neg_mean_squared_error", cv = 5)
    rmse_scores  = np.sqrt(-1 * neg_mse_scores)
    avg_rmse = np.mean(rmse_scores)
    print('##### ',model.__class__.__name__ , ' #####')
    print(' 5 교차 검증의 평균 RMSE : {0:.3f} '.format(avg_rmse))
models = [dt_reg, rf_reg, gb_reg, xgb_reg, lgb_reg]
for model in models:
    get_model_cv_prediction(model, X_data, y_target)
# 아래는 선형회귀 및 회귀트리의 회귀함수 시각화 결과
bostonDF_sample = bostonDF[['RM','PRICE']]
bostonDF_sample = bostonDF_sample.sample(n=100,random_state=0)
print(bostonDF_sample.shape)
plt.figure()
plt.scatter(bostonDF_sample.RM , bostonDF_sample.PRICE,c="darkorange")
lr_reg = LinearRegression()
rf_reg2 = DecisionTreeRegressor(max_depth=2)
rf_reg7 = DecisionTreeRegressor(max_depth=7)
X_test = np.arange(4.5, 8.5, 0.04).reshape(-1, 1)
X_feature = bostonDF_sample['RM'].values.reshape(-1,1)
y_target = bostonDF_sample['PRICE'].values.reshape(-1,1)
lr_reg.fit(X_feature, y_target)
rf_reg2.fit(X_feature, y_target)
rf_reg7.fit(X_feature, y_target)
pred_lr = lr_reg.predict(X_test)
pred_rf2 = rf_reg2.predict(X_test)
pred_rf7 = rf_reg7.predict(X_test)
fig , (ax1, ax2, ax3) = plt.subplots(figsize=(14,4), ncols=3)
ax1.set_title('Linear Regression')
ax1.scatter(bostonDF_sample.RM, bostonDF_sample.PRICE, c="darkorange")
ax1.plot(X_test, pred_lr,label="linear", linewidth=2 )
ax2.set_title('Decision Tree Regression: \n max_depth=2')
ax2.scatter(bostonDF_sample.RM, bostonDF_sample.PRICE, c="darkorange")
ax2.plot(X_test, pred_rf2, label="max_depth:2", linewidth=2 )
ax3.set_title('Decision Tree Regression: \n max_depth=7')
ax3.scatter(bostonDF_sample.RM, bostonDF_sample.PRICE, c="darkorange")
ax3.plot(X_test, pred_rf7, label="max_depth:7", linewidth=2)