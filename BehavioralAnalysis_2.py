# 3 - 1. 견고한 데이터 분석 - 결측 데이터 처리
# 결측 데이터가 존재하는 행을 제거하여 분석하면 결측값이 없는 표본에 데이터가 편향됨 따라서 결측값을 진단하고 분류하여 적절한 처리를 해야함

# %% 결측값 현황 시각화
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer
from statsmodels.formula.api import ols
from statsmodels.imputation import mice

data_df = pd.read_csv("DataSet/chap6-available_data.csv", encoding = "utf-8-sig")
data_supp_df = pd.read_csv("DataSet/chap6-available_data_supp.csv", encoding = "utf-8-sig")
data_df['gender'] = pd.Categorical(data_df.gender, categories=['M','F'])
data_df['state'] = pd.Categorical(data_df.state, categories=['A','B','C'])
def md_pattern(dat_df):
    # 원본 변경 방지
    dat_df = dat_df.copy()

    # 결측치가 있는 컬럼만 추출
    miss_cols = [col for col in dat_df.columns if dat_df[col].isnull().any()]

    # 결측치가 전혀 없는 경우 처리
    if not miss_cols:
        print("\n결측치가 있는 컬럼이 없습니다.")
        return

    miss_df = dat_df[miss_cols]

    # 변수별 결측치 개수
    print(miss_df.isnull().sum())

    # 결측 패턴별 빈도
    pattern_counts = (
        miss_df.isnull()
        .value_counts(dropna=False)
        .rename("count")
        .reset_index()
    )
    print(pattern_counts)
md_pattern(data_df)
md_pattern(data_supp_df)
# md.pattern은 기본적으로 R의 기능이나 파이썬에서도 구현 가능,변수별 결측값의 개수와 결측값 존재 패턴을 확인할 수 있음

# %% 결측값의 양 확인
min_data_df = data_df.copy()
min_data_df.neuro = np.where(min_data_df.neuro.isna(), min_data_df.neuro.min(), min_data_df.neuro)
max_data_df = data_df.copy()
max_data_df.neuro = np.where(max_data_df.neuro.isna(), max_data_df.neuro.max(), max_data_df.neuro)
print(ols("bkg_amt~neuro", data=min_data_df).fit().summary())
print(ols("bkg_amt~neuro", data=max_data_df).fit().summary())
# 결측값이 존재하는 각 변수를 최솟값/최댓값으로 대체한 데이터 프레임을 생성, 3개 데이터 프레핌에 대하여 관심 변수 대상 회귀 분석을 진행한 뒤
# 회귀계수를 비교, 계수의 차이가 미미하면 결측값을 삭제해도 무방한 양이며 차이가 크면 결측값이 데이터에 미치는 영향이 큰 경우임

# %% 결측값의 상관관계 확인
tacoma_df = pd.read_csv("DataSet/chap6-tacoma.csv", encoding = "utf-8-sig")
tampa_df = pd.read_csv("DataSet/chap6-tampa.csv", encoding = "utf-8-sig")
md_pattern(tacoma_df)
md_pattern(tampa_df)
# 실제 데이터에서는 md_pattern만 확인하여 상관관계 여부 판단
# 상관관계 분석은 고차원 데이터에서는 오래 걸리고 가독성 떨어지기 때문
tampa_miss_df = tampa_df.copy().drop(['ID'], axis=1).isna()
tampa_cor = tampa_miss_df.corr()
sns.heatmap(tampa_cor, annot=True, vmin=-0.05, vmax=1, cmap="YlGnBu")
plt.show() # 상관관계가 있는 경우
tacoma_miss_df = tacoma_df.copy().drop(['ID'], axis=1).isna()
tacoma_cor = tacoma_miss_df.corr()
sns.heatmap(tacoma_cor, annot=True, vmin=-0.05, vmax=1, cmap="YlGnBu")
plt.show() # 상관관계가 없는 경우
# 데이터 수집 과정/업무 규칙/휴먼 에러의 영향에 따라 결측값의 존재 여부 사이에 상관관계가 존재할 수 있음, 이때는 변수를 일일이 분석하기보다는
# 상관관계가 있는 변수를 묶어서 분석하는 것이 더 효율적이므로 결측값의 상관관계를 미리 확인할 필요가 있음, 결측값의 존재 패턴을 시각화 했을때
# 모든 변수가 결측인 경우가 매우 적고 결측값의 존재패턴이 무작위적으로 관찰됨에 더해서 여러 변수가 동시에 결측일 확률이 각 변수가 결측일 확률
# 의 곲과 유사하거나 여러 변수가 동시에 결측일 확률 이 그 중 일부 변수가 결측일 확률보다 작으면 변수의 결측값 사이에 완전히 상관관계가 없음

# %% 결측값의 요인 확인
# 결측은 단순한 빈칸이 아니며 분석 결과를 왜곡할 수 있음, 현재 데이터에 어떻게 빈칸이 뚫려 있는가를 확인하지 않고 단순 행 삭제/평균 대체 시
# 분석결과에 편향이 발생할 수 있음, 따라서 결측값의 발생 요인 분석이 중요하며 루빈의 분류에 따라 아래 3가지 케이스로 발생 요인을 분류 가능
# MCAR (완전 무작위 결측) : 결측 여부가 데이터셋에서 관측된 값 및 결측이 발생한 변수 자기 자신의 값 모두에 독립적 = 결측이 발생한 이유가
# 데이터 내 어떤 값과도 무관, 실수로 문항을 빠뜨리거나 전산 오류로 데이터가 누락되는 등 완전 무작위적으로 결측이 발생하는 경우
# MAR (무작위 결측) : 결측 여부가 데이터셋에서 관측된 값에 의존하나 결측이 발생한 변수 자기 자신의 값에는 독립적 = 결측의 원인이 이미 관측
# 된 다른 변수들로 설명 가능한 상황, 관측된 데이터만으로 결측값의 분포를 추정하여 대체 거눙
# MNAR (비무작위 결측) : 결측 여부가 데이터셋에서 관측된 값 및 결측이 발생한 변수 자기 자신의 값 모두에 의존 = 관측된 모든 정보를 통제하고
# 나서도 여전히 자기 자신의 결측값 자체가 결측 여부에 영향을 미침
data_df['md_extra'] = data_df['extra'].isnull().astype(float)
md_extra_mod = smf.logit('md_extra~age+open+neuro+gender+state+bkg_amt', data=data_df)
md_extra_mod.fit().summary() # 완전 무작위 결측 가능성 높음
data_df['md_state'] = data_df['state'].isnull().astype(float)
md_state_mod =smf.logit('md_state~age+open+extra+neuro+gender+bkg_amt', data=data_df)
md_state_mod.fit(disp=0).summary() # 무작위 결측 가능성 높음
data_df['md_neuro'] = data_df['neuro'].isnull().astype(float)
md_neuro_mod =smf.logit('md_neuro~age+open+extra+state+gender+bkg_amt', data=data_df)
md_neuro_mod.fit(disp=0).summary() # 비무작위 결측 가능성 높음
# 결측값의 요인은 우선 결측값이 존재하는 특정 변수에 대해 결측 여부를 표시하는 지시변수를 만든 다음 -> 결측 지시 변수를 종속 변수로 두고 다
# 른 관측가능한 변수를 독립 변수로 두어 로지스틱 회귀를 실행한 다음 -> 지시 변수를 유의하게 설명하는 독립 변수가 있으면 무작위 결측 가능성을
# 높게 판단하고 없으면 완전 무작위 결측 가능성을 높게 판단, 이때 도메인 지식을 고려하여 결측값 자체가 결측을 유발했을 개연성이 이론적으로 크
# 거나 종속 변수를 강하게 설명하는 변수가 다이어그램 상에서 종속 변수에 대한 직접적인 요인일 때 비무작위 결측 가능성을 높게 판단
# 결측 요인의 스펙트럼 : 일반적으로는 결측 요인을 결정적으로 판정할 수는 없으며 완전한 확률론적 극단인 완전 무작위 결측과 완전한 결정론적 극
# 단인 임계값 기반 컷오프(변수의 값이 특정 값 이상/이하인 경우 모두 삭제) 에 존재할 뿐임, 이때 임계값 기반 컷오프가 사용되었을 경우에만 결정
# 론적 비무작위 결측으로 판단할 수 있으며 실제로는 결측 지시 변수 회귀 결과/자료 수집 과정에 대한 맥락 정보/선행연구/비즈니스 규칙을 복합적으
# 로 고려하여 판단

# %% 결측 데이터 처리 - MICE
# 단일 대체법의 한계를 극복하기 위해 결측치를 여러 개의 그럴듯한 값으로 반복대체하여 결측값의 불확실성을 통계적으로 반영하는 다중 대체법 사용
# MICE(Multivariate Imputation by Chained Equations) : 가장 많이 사용되는 다중 대체법이며 일반적으로 예측 평균 매칭(PMM) 기반, 결
# 측 변수 각각에 대해 평균값 등 임시값 사용하여 대체 수행 -> 특정 결측 변수를 종속 변수로 두고 나머지 변수들로 회귀 모델 학습 -> 회귀 모델
# 예측값에 무작위 변동을 더한 값을 결측값으로 사용 -> 결측행의 예측값과 가장 가까운 예측값을 가진 관측 행 k개 선정 -> 해당 행 중 하나를 무
# 작위로 뽑아 그 행의 실제 관측값을 대체값으로 사용 -> 모든 결측 변수에 대해 반복, 예측값이 일정 값으로 수렴하면 종료 -> 전체 과정을 여러번
# 반복하여 여러개의 데이터셋 생성 후 루빈의 규칙 사용해 하나의 데이터프레임으로 통합, MCAR 및 MAR 가정 변수에 대해 유용하게 사용 가능
gender_dummies = pd.get_dummies(data_df.gender, prefix='gender')
data_df =  pd.concat([data_df, gender_dummies], axis=1)
data_df.gender_F = np.where(data_df.gender.isna(), float('NaN'), data_df.gender_F)
data_df.gender_M = np.where(data_df.gender.isna(), float('NaN'), data_df.gender_M)
data_df =  data_df.drop(['gender'], axis=1)
state_dummies = pd.get_dummies(data_df.state, prefix='state')
data_df = pd.concat([data_df, state_dummies], axis=1)
data_df.state_A = np.where(data_df.state.isna(), float('NaN'), data_df.state_A)
data_df.state_B = np.where(data_df.state.isna(), float('NaN'), data_df.state_B)
data_df.state_C = np.where(data_df.state.isna(), float('NaN'), data_df.state_C)
data_df = data_df.drop(['state'], axis=1)
# MICE는 범주형 범주 처리가 불가능하므로 원핫인코딩 실행
MI_data_df = mice.MICEData(data_df)
fit = mice.MICE(model_formula='bkg_amt ~ age + open + extra + neuro + gender_M + gender_F + state_A +'
                              ' state_B + state_C', model_class=sm.OLS, data=MI_data_df)
MI_summ = fit.fit(n_imputations=20).summary()
print(MI_summ)
# 결측 데이터가 존재하는 변수와 관계가 있으나 회귀 모델에 포함시키기에는 어려운 변수를 대체 변수로 추가할 수 있음
augmented_data_df = pd.concat([data_df, data_supp_df], axis=1)
MI_data_aux_df = mice.MICEData(augmented_data_df)
fit = mice.MICE(model_formula='bkg_amt ~ age + open + extra + neuro + \
                gender_M + gender_F + state_A + state_B + state_C',
                model_class=sm.OLS, data=MI_data_aux_df)
MI_aux_summ = fit.fit(n_imputations=20).summary()
print(MI_aux_summ)
# 단순 대체 및 선형회귀 기반 MICE와 비교해서 원래 변수 분포를 더 잘 보존하고, 표준오차의 과소추정 현상을 방지하며, 데이터의 정규성을 가정할
# 필요가 없고, 원본 데이터 범위 밖의 값이 나올 가능성이 적으며 변수 간 상관 관계가 보존되는 장점이 있으나, 속도가 느리고 대규모 데이터에 부
# 적합하며, 결측치가 많은 경우 정확도가 떨어지고, 보간법이므로 인간이 보기에 터무니 없는 값이 선정될 수 있음, MNAR 변수에 대응 불가

# %% 결측 데이터 처리 - MissForest
# MICE의 단점을 보완하는 것이 MissForest, 배깅(원본 데이터에서 복원 추출로 여러 개의 부분 데이터셋을 만들고 그 결과값을 통합) 및 OOB(배
# 깅에서 뽑히지 않은 데이터는 자동으로 검증 세트로 설정) 기반의 랜던 포레스트 기법 사용, 결측치를 임시 값으로 초기 대체 후 결측 비율이 낮은
# 변수부터 처리 순서를 정렬 -> 특정 결측 변수를 종속 변수로 두고 나머지 변수들을 독립변수로 두어 랜덤 포레스트 학습하여 예측값을 대체값으로
# 사용 -> 모든 결측 변수에 대해 반복 -> 예측값이 수렴할때 까지 전체 과정 반복, 수렴 여부는 NMSE의 증가 여부로 판단
def detect_ohe_columns(df):
    ohe_cols = []
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            ohe_cols.append(col)
    return ohe_cols
ohe_columns = detect_ohe_columns(data_df) # 원핫 인코딩된 컬럼 사전 감지
rf_estimator = RandomForestRegressor(
    n_estimators=100,
    max_features="sqrt",     # 트리 학습 시 사용할 변수 수: sqrt(n_features)
    random_state=42,
    n_jobs=-1                # 병렬 처리 (CPU 전체 사용)
)
missforest_imputer = IterativeImputer(
    estimator=rf_estimator,
    max_iter=10,
    tol=1e-3,
    imputation_order="ascending",   # 결측 비율 낮은 변수부터 처리
    random_state=42,
    verbose=2                        # 사이클별 수렴 로그 출력
)
imputed_array = missforest_imputer.fit_transform(data_df)
data_imputed_df = pd.DataFrame(
    imputed_array,
    columns=data_df.columns,
    index=data_df.index
)
data_imputed_df[ohe_columns] = (
    data_imputed_df[ohe_columns]
    .clip(0, 1)   # 혹시 범위 밖으로 나간 값 클리핑
    .round()      # 0.5 기준 반올림 → 0 또는 1
    .astype(int)
) # 랜덤 포레스트 결과값을 다시 원핫 인코딩으로 환원
missing_after = data_imputed_df.isnull().sum().sum()
print(f"\n[대체 후 잔여 결측치 수]: {missing_after}")
continuous_columns = [c for c in data_df.columns if c not in ohe_columns]
if continuous_columns:
    print("\n[연속형 변수 기초 통계량 비교 (대체 전 → 후)]")
    stats_before = data_df[continuous_columns].describe().T[["mean", "std"]]
    stats_after  = data_imputed_df[continuous_columns].describe().T[["mean", "std"]]
    comparison = stats_before.join(stats_after, lsuffix="_before", rsuffix="_after")
    print(comparison.round(4))
# 랜덤 포레스트의 앙상블 특성으로 높은 결측률에도 안정적이고, 트리 기반이므로 이상치에 강건하며, 비선형 관계와 교호작용 자동 포착하고, 병렬처
# 리를 지원하여 속도가 더욱 빠르지만 통계적 불확실성 정량화가 미흡하여 통계적 정확성은 더 떨어지고 여전히 계산 비용이 크며 MNAR 에 대응 불가
# 일반적으로 머신러닝 전처리 용도로는 MissForest, 통계적 분석 용도로는 MICE가 더 적합함, 계산 비용 문제를 해결하기 위해 관측값 기반 K-NN
# 군집화를 통해 결측값을 추정할 수 있으나 고차원 데이터에서 거리가 무의미해지는 차원의 저주 문제가 존재해 결측률이 높을 시 안정성이 크게 떨어
# 지고 통계적 불확실성이 아예 반영되지 않는다.

# %% 결측 데이터 처리 - EM(기댓값 최대화)
# 결측값이 존재하는 변수가 특정 분포를 따른다면 불필요한 계산 비용없이 변수를 특정 분포에 적합시킨다음 대체값을 분포에서 찾을 수 있음 = 확률
# 적 대체법, 예를들어 변수가 다변량 정규 분포를 따른 다고 가정 시 다변량 정규분포의 조건부 분포 공식을 이용하여 조건부 기댓값 계산 가능, 해
# 당 값으로 임시 대체한 다음 다시 모수를 추정하여 기댓값 계산하기를 반복, 결측값의 기댓값이 안정화되면 종료
def impute_em(X: np.ndarray, max_iter: int = 3000, eps: float = 1e-8) -> dict:
    n, p = X.shape
    # 관측 여부 마스크: True = 관측됨
    C = ~np.isnan(X)
    # 각 행의 결측(M_i) / 관측(O_i) 변수 인덱스 사전 계산
    col_idx = np.arange(p)
    M_idx = [col_idx[~C[i]] for i in range(n)]   # 결측 변수 인덱스
    O_idx = [col_idx[C[i]]  for i in range(n)]   # 관측 변수 인덱스

    # 초기화
    mu = np.nanmean(X, axis=0)
    complete_rows = np.where(C.all(axis=1))[0]
    if len(complete_rows) >= 2:
        Sigma = np.cov(X[complete_rows].T)
    else:
        Sigma = np.diag(np.nanvar(X, axis=0))

    X_tilde = X.copy()

    # EM 반복
    for iteration in range(1, max_iter + 1):
        # E-step
        S_accum = np.zeros((p, p))   # 조건부 분산 누적 행렬

        for i in range(n):
            mi, oi = M_idx[i], O_idx[i]
            if len(mi) == 0:         # 결측 없는 행은 건너뜀
                continue

            # 공분산 블록 분해
            Sigma_MM = Sigma[np.ix_(mi, mi)]
            Sigma_MO = Sigma[np.ix_(mi, oi)]
            Sigma_OO = Sigma[np.ix_(oi, oi)]

            # 조건부 평균: E[X_miss | X_obs]
            Sigma_OO_inv = np.linalg.solve(Sigma_OO, np.eye(len(oi)))
            X_tilde[i, mi] = (
                mu[mi]
                + Sigma_MO @ Sigma_OO_inv @ (X_tilde[i, oi] - mu[oi])
            )

            # 조건부 분산: Var[X_miss | X_obs] — M-step 보정용
            Sigma_MM_given_O = Sigma_MM - Sigma_MO @ Sigma_OO_inv @ Sigma_MO.T
            contrib = np.zeros((p, p))
            contrib[np.ix_(mi, mi)] = Sigma_MM_given_O
            S_accum += contrib

        # M-step
        mu_new    = X_tilde.mean(axis=0)
        Sigma_new = (X_tilde.T @ X_tilde) / n \
                    - np.outer(mu_new, mu_new) \
                    + S_accum / n          # 조건부 분산 보정항

        # 수렴 판단
        delta_mu    = np.linalg.norm(mu_new - mu)
        delta_Sigma = np.linalg.norm(Sigma_new - Sigma, ord=2)
        mu, Sigma = mu_new, Sigma_new

        if delta_mu < eps and delta_Sigma < eps:
            print(f"수렴 완료: {iteration}회 반복")
            break
    else:
        print(f"경고: 최대 반복 횟수({max_iter})에 도달했으나 미수렴")

    return {
        "X_imputed" : X_tilde,
        "mu"        : mu,
        "Sigma"     : Sigma,
        "n_iter"    : iteration
    }
def detect_ohe_columns(df):
    return [c for c in df.columns
            if set(df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
ohe_cols  = detect_ohe_columns(data_df)
cont_cols = [c for c in data_df.columns if c not in ohe_cols]
print(f"EM 적용 대상 연속형 변수: {len(cont_cols)}개")
print(f"EM 제외 OHE 변수       : {len(ohe_cols)}개")
# EM 대체 수행
X_cont = data_df[cont_cols].to_numpy(dtype=float)
result = impute_em(X_cont, max_iter=3000, eps=1e-8)
# 결과를 DataFrame으로 복원
data_imputed_df = data_df.copy()
data_imputed_df[cont_cols] = result["X_imputed"]
# OHE 컬럼은 최빈값(0 또는 1)으로 별도 대체
for col in ohe_cols:
    mode_val = data_df[col].mode()[0]
    data_imputed_df[col] = data_df[col].fillna(mode_val)
# 결과 검증
print(f"\n[대체 후 잔여 결측치]: {data_imputed_df.isnull().sum().sum()}")
print("\n[수렴된 μ vs 관측 평균]")
comparison = pd.DataFrame({
    "observed_mean" : np.nanmean(X_cont, axis=0),
    "em_mu"         : result["mu"]
}, index=cont_cols)
print(comparison.round(4))
print(f"\n총 반복 횟수: {result['n_iter']}")
# MICE나 MissForest와 비교해서 변수의 분포가 확실할 시 계산이 적고 대규모 데이터에 유리하며, 초기값이 같으면 항상 동일한 결과로 수렴하는
# 재현성을 갖추고 있으며, 앞의 두 알고리즘과 달리 이론적으로 수렴 여부를 판단할 수 있으나 단일 대체이므로 대체된 값의 분산이 과소 추정되는
# 문제가 있다. 이를 보완하기 위해 베이지안 대체 등 다른 기법을 사용할 수 있으나 EM 알고리즘 보다 계산 비용이 크다.


# %% 3 - 2. 견고한 데이터 분석 - 부트스트랩
# 어떤 데이터셋에 대하여 평균이나 애드혹 통계(문제 해결/의사결정 등 특정 목적을 위해 일회성으로 수집하는 통계값 예를 들어 특정 시점에서 특정
# 변수가 어떤 값 이상인 케이스 등)을 도출하고자 할 수 있음, 이때 추출된 표본이 모두 이상적일 수는 없으며 모집단으로부터 충분한 양과 질을 가
# 진 표본을 추출하는 것 자체가 현실적으로 어려운 경우가 많음, 따라서 표본의 크기가 작으면서 이상치가 존재하는 비이상적인 표본도 존재할 수 있
# 음, 이런 데이터에서 단순하게 이상치를 제외하는 것은 바람직하지 않으며 단일 통계값만 사용한다면 데이터의 분포를 무시하게 되고 신뢰구간을 추
# 정한다면 실제로 존재할 수 없는 값이 도출될 가능성이 있음
# 부트스트랩 사용 시 해당 문제 해결 가능 : 표본 자체를 모집단처럼 취급해 반복 재표집함으로써 통계량의 분포를 추정, 표본이 모집단을 잘 대표한
# 다면 표본에서 재표집한 결과는 모집단에서 바로 표집한 것에 근사시킬 수 있기 때문

# %% 부트스트랩 기반 통계값 신뢰구간 추정
# 표본 수가 적고 이상치가 존재하는 n개의 원표본에 대해 : 원표본으로부터 복원추출하여 크기가 n인 서로 다른 표본을 무수히 많이 만든 후 -> 각
# 표본에 대하여 원하는 통계값을 계산한 뒤 -> 통계값의 분포를 통해 통계값 추정의 불확실설(표준오차/신뢰구간 등)을 추정
import statsmodels.api as sm
import statsmodels.stats.outliers_influence as st_inf
times = [2,2,3,5,6,9,10,47,61,413]
experience = [11,17,18,1,10,4,6,3,8,0]
data_boot_df = pd.DataFrame(
    {'times': times,
    'experience': experience})
# 부트스트랩으로 표본 평균 신뢰구간 추정
res_boot_sim = []
B = 2000
N = len(data_boot_df)
for i in range(B):
    boot_df = data_boot_df.sample(N, replace=True)
    M = np.mean(boot_df.times)
    res_boot_sim.append(M)
LL_b = np.quantile(res_boot_sim, 0.025)
UL_b = np.quantile(res_boot_sim, 0.975)
print("LL_b = ", LL_b)
print("UL_b = ", UL_b)
# 부트스트랩으로 애드혹 통계 신뢰구간 추정
promise_lst = []
B = 2000
N = len(data_boot_df)
for i in range(B):
    boot_df = data_boot_df.sample(N, replace = True)
    above180 =  len(boot_df[boot_df.times >= 180]) / N
    promise_lst.append(above180)
LL_b = np.quantile(promise_lst, 0.025)
UL_b = np.quantile(promise_lst, 0.975)
print("LL_b = ", LL_b)
print("UL_b = ",UL_b)
# 데이터의 분포에 대한 사전 가정이나 중심 극한 정리에 의존하지 않고도 표본 크기나 형태에 무관하게 합리적인 신뢰구간을 도출 가능하며 원표본의
# 범위를 벗어나는 수치가 나올 가능성을 차단할 수 있음, 다만 표본이 편향없이 모집단을 적절하게 대표한다는 가정을 만족해야하며 결측치와 오류성
# 이상치는 사전에 처리한 뒤 추정해야함

# %% 부트스트랩 기반 회귀분석
# 표본 수가 적고 이상치가 존재하는 n개의 원표본의 변수 X와 Y에 대한 회귀분석을 수행할때 단순 최소제곱법 사용 시 특정 이상치의 회귀선에 대한
# 기여도가 지나치게 커지며 회귀선이 왜곡됨, 따라서 회귀계수의 신뢰구간을 추정 시 임계값(보통 0)이 포함되어 결론을 내릴 수 없게 될 가능성 높
# 음, 또한 부트스트랩을 최소제곱법에 바로 결합하면 이상치가 한 표본에 여러 번 들어가 이상치 비율이 지나치게 높은 표본 생성될 수 있음
# MM-추정 기반 로버스트 부트스트랩 기법으로 해결 : 복원추출로 만든 무수한 부트스트랩 표본들에서 회귀계수를 계산할때, MM-추정과 같은 강건 회
# 귀를 사용, 이때 잔차의 크기에 Threshold를 정해 그 이하로 크기를 제한한 다음 회귀계수를 추정한 다음(S-추정) 다시 회귀선에서 거리가 멀어질
# 수록 가중치를 낮추어 다시 한번 회귀계수를 추정하기 때문에(M-추정), 이상치가 많아도 강건하여 정확한 회귀계수 신뢰 구간 추정 가능
n_iterations = 4000  # 부트스트랩 반복 횟수
boot_coefs = []  # 기울기(coefficient)를 저장할 리스트
for i in range(n_iterations):
    # (1) 복원 추출로 가상 표본 생성
    sample = data_boot_df.sample(n=len(data_boot_df), replace=True)
    X_sample = sm.add_constant(sample['experience'])  # 절편항 추가
    y_sample = sample['times']
    try:
        # (2) MM-추정의 핵심인 TukeyBiweight 손실함수를 사용한 강건 회귀
        # statsmodels의 RLM은 기본적으로 M-추정 계열이며, 스케일 추정 방식을 지정하여 MM-추정과 유사하게 동작 가능
        model = sm.RLM(y_sample, X_sample, M=sm.robust.norms.TukeyBiweight())
        results = model.fit()
        # 'experience'의 기울기 저장
        boot_coefs.append(results.params['experience'])
    except:
        # 드물게 샘플링 과정에서 수렴하지 않는 경우 방지
        continue
boot_coefs = np.array(boot_coefs)
mean_coef = np.mean(boot_coefs)
ci_lower = np.percentile(boot_coefs, 2.5)
ci_upper = np.percentile(boot_coefs, 97.5) # 95% 신뢰구간
print("-" * 30)
print(f"분석 결과 (변수: experience -> times)")
print(f"평균 기울기(Coefficient): {mean_coef:.4f}")
print(f"95% 신뢰구간: [{ci_lower:.4f}, {ci_upper:.4f}]")
print("-" * 30) # 결과 집계
plt.figure(figsize=(10, 6))
plt.hist(boot_coefs, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(mean_coef, color='red', linestyle='--', label=f'Mean: {mean_coef:.2f}')
plt.axvline(ci_lower, color='orange', linestyle=':', label='95% CI')
plt.axvline(ci_upper, color='orange', linestyle=':')
plt.title('Bootstrap Distribution of Regression Coefficient (MM-estimation)')
plt.xlabel('Coefficient Value (Experience)')
plt.ylabel('Frequency')
plt.legend()
plt.show() # 시각화
# 평균 회귀 계수는 -4.76이나 시각화 결과 신뢰구간이 0을 포함하고 있으며 회귀계수의 분포가 양극화되어있음, 이는 숙련도가 늘어나도 처리시간이
# 별로 줄어들지 않는 근로자와 획기적으로 줄어드는 근로자로 양극화 되어있음을 의미하며 숙련도가 증가할때마다 평균 -4.76 만큼 처리시간이 줄어
# 든다고 해석해서는 안 됨, 회귀계수 평균보다는 회귀계수의 분포가 더 중요하며 데이터 세분화 및 이상치 재확인을 통해 데이터를 더 정제하거나 표
# 본 개수를 늘리거나 그 원인을 도메인 지식을 통해 분석하는 것이 필요

# %% 부트스트랩의 사용조건, 최적화
# 우선 표본에서 점추정을 함에 있어서 전통적인 방법으로 유의한 점 추정치를 산출할 수 있을때는 부트스트랩이 무의미함, 데이터가 적거나 분포상에
# 여러개의 피크가 있거나 비대칭인 경우 등 비일반적인 데이터의 경우 부트스트랩을 통한 신뢰구간 추정이 유의미함, 마찬가지로 회귀계수가 경계 또
# 는 임계점에 가까워 명확하지 않은 경우에도 부트스트랩을 통한 신뢰구간 추정이 유의미함, 한편 신뢰구간 추정을 함에 있어서 영향점(해당 데이터
# 를 삭제했을때 회귀가 크게 변하는 데이터)가 없고 잔차의 분포가 정규성을 가지는 경우 부트스트랩이 무의미함, 영향점을 측정하는 쿡의 거리가 1
# 이상이거나 잔차의 분포가 정규분포를 따르지 않는 경우 부트스트랩을 통한 신뢰구간 추정이 유의미함
lin_mod = ols("times~experience", data=data_boot_df).fit()
print(lin_mod.summary())
CD = st_inf.OLSInfluence(lin_mod).summary_frame()['cooks_d']
print(CD[CD > 1]) # 영향점 체크
res_df = lin_mod.resid
sns.kdeplot(res_df)
fig = sm.qqplot(res_df, line='s')
plt.show() # 잔차의 정규성 확인
# 파이썬에서 부트스트랩의 경우 넘파이 기반으로 구현할 시 일일히 무작위 추출하는 대신 벡터화를 통해 한번에 처리하고, 병렬 연산 및 메모리뷰 및
# 행렬곱을 통한 효율적인 데이터 처리가 가능하므로 퍼포먼스가 더욱 향상됨, 그러나 이 경우 메모리를 적극적으로 사용하므로 대용량 데이터 처리 시
# 청크 별로 나누어 처리하는 방법도 고려해야함
import time
np.random.seed(42)
data = np.random.normal(loc=30, scale=5, size=50)
n_iterations = 100000
n_size = len(data)
start_time = time.time()
# 한 번에 모든 인덱스를 무작위로 추출 (10만 행 x 50열 행렬), replace=True 옵션으로 복원 추출 수행
boot_indices = np.random.randint(0, n_size, size=(n_iterations, n_size))
# 인덱스 행렬을 실제 데이터 값으로 매핑
boot_samples = data[boot_indices]
# 행(axis=1) 방향으로 평균을 한 번에 계산, 결과값은 10만 개의 가상 표본 평균들이 담긴 배열이 됨
boot_means = np.mean(boot_samples, axis=1)
mean_estimate = np.mean(boot_means)
ci_lower = np.percentile(boot_means, 2.5)
ci_upper = np.percentile(boot_means, 97.5)
end_time = time.time()
print(f"--- 분석 결과 (반복 횟수: {n_iterations:,}회) ---")
print(f"추정 평균치: {mean_estimate:.4f}")
print(f"95% 신뢰구간: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"연산 소요 시간: {end_time - start_time:.4f} 초") # 결과 집계