# 4 - 2. 실험 설계와 분석 - 층화 무작위 배정 기반
# 단순 무작위 배정 시 평균적으로는 대조군과 실험군이 비슷해지나 이는 표본이 충분할 때 각 집단별 기댓값이 유사해진다는 의미이고 실제 각 표본에
# 서는 우연에 의해 나이/성별/이용 빈도 같은 중요한 공변량이 한쪽으로 쏠릴 수 있음, 이는 표본 수가 작을수록 그리고 공변량이 결과변수에 강하게
# 영향을 미칠수록 더 심각해짐, 층화 무작위 배정은 전체 표본을 먼저 공변량에 따라 층 또는 소집단으로 나눈 다음 각 층 안에서 독립적으로 무작위
# 배정을 진행하여 집단간 균형을 구조적으로 보장함, 따라서 데이터의 각 층간 분산은 최소화되고 각 층내의 분산만 영향을 미치게 되어 개입의 효과
# 를 더 정확하게 측정할 수 있으며 같은 표본수로 더 높은 검정력을 얻을 수 있고 동일한 검정력을 위한 표본은 더 적게 필요함
import pandas as pd
import numpy as np
import random
import itertools
from scipy.spatial import distance_matrix
from scipy.stats import f_oneway, chi2_contingency
from statsmodels.formula.api import ols
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from statsmodels.stats.weightstats import ttest_ind
from scipy.stats import ttest_1samp
hist_data_df = pd.read_csv('DataSet/chap9-historical_data.csv', encoding="utf-8-sig")
exp_data_df = pd.read_csv('DataSet/chap9-experimental_data.csv', encoding="utf-8-sig")
# 데이터 로드
for df in [hist_data_df, exp_data_df]:
    df['tier'] = pd.Categorical(df['tier'], categories=[3, 2, 1], ordered=True)
    df['ID']   = df['ID'].astype(str)
# 범주형 변수 지정 및 ID 문자열화
hist_data_df.head()
# 해당 데이터셋에 개입하는 처치는 총 2종류이며, 목표변수는 BPDay = 일일 예약 수익임

# %% 층화 무작위 배정의 원리
# 층화 배정의 출발점은 서로 비슷한 관찰 단위끼리 묶는 것이며 비슷하다는 개념을 컴퓨터가 계산할 수 있으려면 수치화해야함, 보통 유클리드 거리를
# 사용하여 각각의 표본을 다차원 공간의 점으로 보고 두 점 사이의 직선 거리가 작을수록 비슷하다고 판단, 그러나 변수들의 단위나 범위가 서로 다르
# 므로 단순히 범위가 큰 변수가 다른 변수의 영향을 압도할 위험성이 생김, 따라서 변수간 정규화를 실시하여 변수간 범위를 통일, 다만 이 경우에도
# 한 범주형 변수의 차이와 한 수치형 변수의 최솟값 - 최댓값간의 차이가 동등한 기여를 하게 되므로 완전히 이상적인 처리는 아님 따라서 도메인 지
# 식을 고려한 보다 정교한 정규화 처리가 필요할 수 있음
def strat_prep_fun(dat_df):
    temp_df = dat_df.copy()
    # ID × tier 기준으로 집계하여 매물 단위 특성 추출
    # tier를 groupby에 포함시키는 이유 : tier는 매물 고유 속성이므로
    # 집계 후에도 범주형 변수로 보존하기 위함이다
    temp_df = temp_df.groupby(['ID', 'tier']).agg(
        sq_ft      = ('sq_ft',       'mean'),
        avg_review = ('avg_review',  'mean'),
        BPday      = ('BPday',       'mean')
    ).dropna().reset_index()
    # 수치형 컬럼과 범주형 컬럼 분리
    num_df = temp_df.loc[:, temp_df.dtypes == 'float64'].copy()
    cat_df = temp_df.loc[:, temp_df.dtypes == 'category'].copy()
    # 수치형 → Min-Max 정규화 ([0, 1] 범위로 스케일 통일)
    # 변수마다 단위가 달라도 거리 계산에서 동등한 비중
    scaler  = MinMaxScaler()
    num_np  = scaler.fit_transform(num_df)
    # 범주형 → 원-핫 인코딩 (tier 3개 수준 → 3차원 이진 벡터)
    enc    = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cat_np = enc.fit_transform(cat_df)
    # 수치형 배열 + 범주형 배열을 열 방향으로 결합
    # 최종 형태 : (N_매물 × 6) 배열
    # 열 구성 : [sq_ft, avg_review, BPday, tier=3, tier=2, tier=1]
    data_np = np.concatenate([num_np, cat_np], axis=1)
    return data_np
# 전처리가 끝나면 N개의 매물 각각에 대해 나머지 N-1개와의 거리를 모두 계산해 N x N 거리 행렬이 생성되며, 이때 대각 원소 즉 자기 자신과의 거
# 리는 매우 큰 값으로 채워 절대 선택되지 않도록 처리한다음, 그리디 알고리즘 방식으로 각 매물에 대해 가장 가까운 이웃들을 비복원 추출하여 블록
# 을 구성, 이때 한 블록에 실험군/대조군의 개수만큼의 데이터가 들어올때까지 추출하고 이미 추출된 데이터는 다시 추출하지 않음, 한 블록이 완성되
# 면 추출되지 않은 한 매물에 대해서 가장 가까운 이웃들을 추출하여 블록을 완성하기를 반복, 모든 블록이 완성되면 각 블록의 원소들을 각각 하나의
# 실험군/대조군에 배정 -> 예를 들어 A, B, E, C, D, F 순서로 거리가 가까운 표본 6개가 있고 실험군/대조군 개수는 3개 일때 첫번째 블록에 A,
# B, E 순서로 3개의 데이터를 추출하여 완성하고 이미 추출된 표본 B는 건너뛴다음 두번째 블록에 C, D, F 순서로 추출하여 완성, 각 블록의 원소
# 3개에 1 대 1 대응으로 하나의 실험군/대조군 배정
def stratified_assgnt_fun(dat_df, K):
    # K의 배수가 되도록 표본 수 조정, 블록 크기가 K이므로 나머지가 있으면 그만큼 무작위로 제외
    remainder = len(dat_df) % K
    if remainder != 0:
        dat_df = dat_df.sample(len(dat_df) - remainder)
    # 전처리 함수가 행 순서를 바꿀 수 있으므로 ID 목록을 미리 추출
    dat_ID = dat_df['ID'].astype(str).tolist()
    # strat_prep_fun으로 모든 행(매물) 쌍 간의 유클리드 거리를 계산해 (N × N) 행렬 생성
    data_np = strat_prep_fun(dat_df)
    N = len(data_np)
    d_mat = distance_matrix(data_np, data_np)
    # 대각 원소는 → N+1 같은 충분히 큰 값으로 채워 선택 불가 처리
    np.fill_diagonal(d_mat, N + 1)
    # match_len : 각 블록에서 대상 데이터 외에 추가로 찾아야 할 이웃 수 = K-1
    # match_idx : np.argpartition의 kth 파라미터용, 자기자신을 제외하고 거리가 몇번째로 작은 이웃까지의 인덱스까지 반환할지
    match_len = K - 1
    match_idx = match_len - 1
    lim = int(N / match_len)  # 최종 블록 수 한도, 실제 블록수 보다 크나 무한반복 방지를 위해 사용
    available_temp = list(range(N))
    matches_lst    = []
    # np.argpartition(d_mat, kth=match_idx) : 각 행에서 거리가 가장 작은 match_idx + 1개의 인덱스를 탐색
    # argsort 같은 함수와 달리 전체를 정렬하지 않기 때문에 훨씬 빠름
    closest = np.argpartition(d_mat, kth=match_idx, axis=1)
    for n in range(N):
        if len(matches_lst) == lim:
            break
        if n not in available_temp:
            continue  # 이미 매칭된 매물은 건너뜀
        # match_lim을 점진적으로 늘려가며 이웃 탐색 범위를 확장
        # available_temp에 남아있는 이웃이 K-1명 모일 때까지 반복
        for match_lim in range(match_idx, N - 1):
            possible_matches = closest[n, :match_lim].tolist()
            matches = list(set(available_temp) & set(possible_matches))
            if len(matches) == match_len:
                # 블록 완성 : 매물 n + 가장 가까운 이웃 K-1개
                matches.append(n)
                matches_lst.append(matches)
                available_temp = [m for m in available_temp if m not in matches]
                break
            else:
                # 탐색 범위를 한 칸 더 넓혀서 재탐색
                closest[n, :] = np.argpartition(d_mat[n, :], kth=match_lim)
    # 블록 내 무작위 배정 : (N/K) × K 형태의 그룹 번호 배열 생성 후, 행(블록)별로 셔플
    # 예) K=3, N/K=3이면 [[0,1,2],[0,1,2],[0,1,2]] → 행별로 섞음
    exp_grps = np.array(list(range(K)) * int(N / K)).reshape(int(N / K), K).tolist()
    for j in exp_grps:
        np.random.shuffle(j)
    # 중첩 리스트를 1차원으로 펼침
    exp_grps_flat    = list(itertools.chain(*exp_grps))
    matches_lst_flat = list(itertools.chain(*matches_lst))
    # 매칭 인덱스 순서에 맞게 그룹 번호 재정렬
    exp_grps_reordered = [x for _, x in sorted(zip(matches_lst_flat, exp_grps_flat))]
    # 그룹 번호 → 그룹 이름 변환
    grp_map = {'0': 'ctrl', '1': 'treat1', '2': 'treat2'}
    assgnt_df = pd.DataFrame({'grp': exp_grps_reordered})
    assgnt_df['grp'] = assgnt_df['grp'].astype(str).map(grp_map)
    assgnt_df['ID']  = dat_ID
    # 원본 데이터프레임에 배정 결과 병합
    return dat_df.merge(assgnt_df, on='ID', how='inner')
random.seed(42)
per = random.sample(range(35), 1)[0] + 1
sample_df = hist_data_df.loc[hist_data_df['period'] == per].sample(300, random_state=42)
stratified_data_df = stratified_assgnt_fun(sample_df, K=3)
print(f"사용된 기간(period) : {per}")
print(f"전체 표본 수        : {len(stratified_data_df)}")
print(f"\n그룹별 배정 결과 :\n{stratified_data_df['grp'].value_counts()}")
print(f"\n샘플 확인 :\n{stratified_data_df[['ID','sq_ft','tier','avg_review','BPday','grp']].head(6)}")

# %% 층화 배정 품질 확인
# 그룹별 연속형 변수 서술 통계량 확인
cont_vars = ['sq_ft', 'avg_review', 'BPday']
print("=== 그룹별 연속형 변수 평균 (서술 통계) ===")
desc = stratified_data_df.groupby('grp')[cont_vars].mean().round(4)
print(desc)
print("\n=== 그룹 간 변수 평균의 표준편차 (작을수록 균형) ===")
for v in cont_vars:
    group_means = stratified_data_df.groupby('grp')[v].mean()
    sd_between  = group_means.std()
    print(f"  {v:>12s} : {sd_between:.4f}")
# 연속형 변수 — 그룹 간 일원 분산분석 : 이때 귀무가설은 그룹 간 해당 변수의 평균 차이가 없다이며 배정이 균형 잡혀 있다면 P-value가 유의하지
# 않아 귀무가설 기각에 실패함
print("\n=== 연속형 변수 One-Way ANOVA ===")
ALPHA = 0.05
for v in cont_vars:
    groups = [g[v].values for _, g in stratified_data_df.groupby('grp')]
    stat, p = f_oneway(*groups)
    balanced = "균형" if p >= ALPHA else "⚠ 불균형 의심"
    print(f"  {v:>12s} | F={stat:.4f}  p={p:.4f}  → {balanced}")
# 범주형 변수 — 그룹 간 tier 분포 카이제곱 검정 : 이때 귀무가설은 그룹 간 tier 분포 비율이 같다이며 마찬가지로 배정이 균형 잡혀 있다면 P-
# value가 유의하지 않아 귀무가설 기각에 실패함
print("\n=== 범주형 변수 (tier) 카이제곱 검정 ===")
tier_ct = pd.crosstab(stratified_data_df['grp'], stratified_data_df['tier'])
print(tier_ct)
chi2, p_tier, dof, _ = chi2_contingency(tier_ct)
balanced_tier = "균형" if p_tier >= ALPHA else "⚠ 불균형 의심"
print(f"\n  Chi2={chi2:.4f}  p={p_tier:.4f}  df={dof}  → {balanced_tier}")

# %% 부트스트랩 시뮬레이션 기반 표본 검정력 확인 및 표본 개수 결정
# 해당 실험의 목표변수 BPDay는 이항 비율이 아닌 OLS 회귀 계수로 공변량이 포함된 회귀 모형의 계수는 단순 이항 비율과 달리 Closed-Form Sol
# ution이 존재하지 않음 따라서 실험을 여러번 실행한 후 탐지 여부를 계측하는 방법으로 검정력을 검증하는 시뮬레이션 기법을 사용해야함, 이때 표
# 본의 개수를 증가시켜가며 검정력을 측정하여 필요 표본수 도출 가능
# 각 실험군에 대한 BPday 선형 회귀 적합
def treat1_metric_fun(dat_df):
    res = ols("BPday ~ sq_ft + tier + avg_review + grp", data=dat_df).fit(disp=0)
    return res.params['grp[T.treat1]']
def treat2_metric_fun(dat_df):
    res = ols("BPday ~ sq_ft + tier + avg_review + grp", data=dat_df).fit(disp=0)
    return res.params['grp[T.treat2]']
# 복원추출로 여러 개의 가상 표본을 만들고 각각 metric_fun을 계산해 경험적 분포의 분위수를 신뢰구간으로 사용
def boot_CI_fun(dat_df, metric_fun, B=100, conf_level=0.9):
    N = len(dat_df)
    coeffs = sorted([
        metric_fun(dat_df.sample(n=N, replace=True))
        for _ in range(B)
    ])
    start_idx = round(B * (1 - conf_level) / 2)
    end_idx   = -round(B * (1 - conf_level) / 2)
    return [coeffs[start_idx], coeffs[end_idx]]
# 단일 실험 의사결정 - Bootstrap CI 하한이 0 초과이면 효과 탐지 성공(1), 아니면 실패(0)
def decision_fun(dat_df, metric_fun, B=100, conf_level=0.9):
    ci = boot_CI_fun(dat_df, metric_fun, B=B, conf_level=conf_level)
    return 1 if ci[0] > 0 else 0
# 단일 시뮬레이션 실험 : 과거 데이터에서 임의 기간 선택 후 Nexp개 표본 추출 -> 층화 배정 실행 후 treat2 그룹에만 eff_size만큼 BPday에
# 가산하여 처치 효과 주입 -> decision_fun으로 효과 탐지 성공 여부 반환
def single_sim_fun(dat_df, metric_fun, Nexp, eff_size, B=100, conf_level=0.9):
    per        = random.sample(range(35), 1)[0] + 1
    sample_df  = dat_df.loc[dat_df['period'] == per].sample(n=Nexp)
    sim_df     = stratified_assgnt_fun(sample_df, K=3)
    sim_df     = sim_df.copy()
    sim_df['BPday'] = np.where(sim_df['grp'] == 'treat2',
                               sim_df['BPday'] + eff_size,
                               sim_df['BPday'])
    return decision_fun(sim_df, metric_fun, B=B, conf_level=conf_level)
# 검정력 시뮬레이션 실험 : Nsim번 단일 시뮬레이션 실험을 실시하여 검정력 집계
def power_sim_fun(dat_df, metric_fun, Nexp, eff_size, Nsim, B=100, conf_level=0.9):
    results = [
        single_sim_fun(dat_df, metric_fun,
                       Nexp=Nexp, eff_size=eff_size,
                       B=B, conf_level=conf_level)
        for _ in range(Nsim)
    ]
    return np.mean(results)
# 표본 크기별 검정력 추정 : treat2의 BPday 효과 크기 2를 기준으로 표본 크기를 변화시키며 검정력이 어떻게 달라지는지 확인
print("=== treat2 검정력 시뮬레이션 ===")
print(f"  효과 크기(eff_size) : 2  |  신뢰수준 : 90%  |  Nsim : 50\n")
for nexp in [1000, 2000, 4000, 8000, 16000]:
    power = power_sim_fun(
        hist_data_df,
        metric_fun  = treat2_metric_fun,
        Nexp        = nexp,
        eff_size    = 2,
        Nsim        = 50,
        B           = 100,
        conf_level  = 0.9
    )
    print(f"  Nexp = {nexp:>4d}  →  추정 검정력 : {power:.2f}")

# %% 실험 결과 분석 - 컴플라이언스 문제
# 이론과는 달리 현실에서는 배정은 됐지만 처치를 실제로 받지 않은 데이터가 존재할 수 있으며 이를 비컴플라이언스 라고 지칭, 비컴플라이언스 데이
# 터가 많아질수록 처치를 실제로 받았는지 여부와 무관하게 배정된 그룹 그대로 분석하는 ITT(Intent-to-Treat) 에서는 처치 효과가 희석됨, 따라
# 서 컴플라이언스 보정을 통해 만약 모든 매물이 처치를 이행했을때 예상되는 효과를 계산할 필요가 있음
exp_data_reg_df = exp_data_df.copy()
exp_data_reg_df['BPday'] = np.where(
    (exp_data_reg_df['compliant'] == 1) & (exp_data_reg_df['grp'] == 'treat2'),
    exp_data_reg_df['BPday'] - 10,
    exp_data_reg_df['BPday']
) # treat2(최소 예약 기간 강제) 집단에 최소 예약 기간 강제로 인한 일일 수익 감소 예상치 -10 반영
# 컴플라이언스 현황 파악 : 분석 전에 처치 이행률을 먼저 확인
print("=== 그룹별 컴플라이언스 비율 ===")
compliance = exp_data_reg_df.groupby('grp').agg(
    n              = ('compliant', 'count'),
    compliant_n    = ('compliant', 'sum'),
    compliance_rate = ('compliant', 'mean')
).round(3)
print(compliance)
# ITT 분석 — 공변량 보정 OLS 회귀
print("\n=== OLS 회귀 결과 (ITT) ===")
ols_res = ols("BPday ~ sq_ft + tier + avg_review + grp",
              data=exp_data_reg_df).fit(disp=0)
print(ols_res.summary())
coef_t1 = ols_res.params.get('grp[T.treat1]', None)
coef_t2 = ols_res.params.get('grp[T.treat2]', None)
p_t1    = ols_res.pvalues.get('grp[T.treat1]', None)
p_t2    = ols_res.pvalues.get('grp[T.treat2]', None)
print(f"\n  treat1 계수: {coef_t1:.4f}  (p={p_t1:.4f})")
print(f"  treat2 계수: {coef_t2:.4f}  (p={p_t2:.4f})")
# ITT 분석 — 실험군 간 BPday 차이에 대한 Bootstrap 신뢰구간 도출
BOOT_B     = 200
CONF_LEVEL = 0.9
print("\n=== Bootstrap 신뢰구간 ===")
ci_t1 = boot_CI_fun(exp_data_reg_df, treat1_metric_fun, B=BOOT_B, conf_level=CONF_LEVEL)
ci_t2 = boot_CI_fun(exp_data_reg_df, treat2_metric_fun, B=BOOT_B, conf_level=CONF_LEVEL)
print(f"  treat1 {int(CONF_LEVEL*100)}% CI: [{ci_t1[0]:.4f}, {ci_t1[1]:.4f}]")
print(f"  treat2 {int(CONF_LEVEL*100)}% CI: [{ci_t2[0]:.4f}, {ci_t2[1]:.4f}]")
# 신뢰구간 해석
for label, ci in [("treat1", ci_t1), ("treat2", ci_t2)]:
    if ci[1] < 0:
        verdict = "유의한 감소 효과"
    elif ci[0] > 0:
        verdict = "유의한 증가 효과"
    else:
        verdict = "유의하지 않음 (0 포함)"
    print(f"  {label}: {verdict}")
# 컴플라이언스 미보정 시 - 단순 평균 비교 (t-검정) : 이때 귀무가설은 ctrl BPday >= treat1 BPday이며 처치의 효과가 있었다면 P-value가
# 유의하여 귀무가설 기각에 성공함
ctrl_bpday   = exp_data_reg_df[exp_data_reg_df['grp'] == 'ctrl']['BPday']
treat1_bpday = exp_data_reg_df[exp_data_reg_df['grp'] == 'treat1']['BPday']
treat2_bpday = exp_data_reg_df[exp_data_reg_df['grp'] == 'treat2']['BPday']
t_stat1, p_ttest1, dof1 = ttest_ind(ctrl_bpday, treat1_bpday, alternative='smaller')
print("\n=== treat1 단측 t-검정 (ctrl < treat1) ===")
print(f"  t 통계량 : {t_stat1:.4f}")
print(f"  p-value  : {p_ttest1:.4f}")
print(f"  자유도   : {dof1:.0f}")
print(f"  결론     : {'treat1 유의한 증가 효과' if p_ttest1 < 0.05 else '유의하지 않음'}")
# 컴플라이언스 보정 시 - 보정 효과 추정치(CACE) 비교 (Bootstrap t-검정)
BOOT_B     = 500
ALPHA      = 0.05
RANDOM_SEED = 42
compliance_rates = (
    exp_data_reg_df.groupby('grp')['compliant']
    .mean()
    .to_dict()
)
cr_t1 = compliance_rates['treat1']
cr_t2 = compliance_rates['treat2']
print(f"treat1 컴플라이언스율: {cr_t1:.4f}")
print(f"treat2 컴플라이언스율: {cr_t2:.4f}")
# Bootstrap 보정 효과 추정치(CACE) 추출
# : 복원추출로 B개의 가상 표본 생성 → 각각 OLS 적합 → ITT 계수 추출 → ITT ÷ compliance_rate로 보정 효과 추정치(CACE) 계산
boot_rng  = np.random.default_rng(RANDOM_SEED)
N         = len(exp_data_reg_df)
cace_t1_boot = []
cace_t2_boot = []
for _ in range(BOOT_B):
    boot_df = exp_data_reg_df.sample(
        n=N, replace=True,
        random_state=int(boot_rng.integers(0, 10**9))
    )
    res = ols("BPday ~ sq_ft + tier + avg_review + grp", data=boot_df).fit(disp=0)
    itt_t1 = res.params.get('grp[T.treat1]', np.nan)
    itt_t2 = res.params.get('grp[T.treat2]', np.nan)
    cace_t1_boot.append(itt_t1 / cr_t1)
    cace_t2_boot.append(itt_t2 / cr_t2)
cace_t1_boot = np.array(cace_t1_boot)
cace_t2_boot = np.array(cace_t2_boot)
# 단측 t-검정 treat1 : 이때 귀무가설은 CACE_treat1 <= 0 이며 대립가설은 CACE_treat1 > 0
t1_stat, t1_p = ttest_1samp(cace_t1_boot, popmean=0, alternative='greater')
# treat2 : 이때 귀무가설은 CACE_treat2 >= 0 이며 대립가설은 CACE_treat2 < 0
t2_stat, t2_p = ttest_1samp(cace_t2_boot, popmean=0, alternative='less')
print("\n=== CACE Bootstrap 분포 기술 통계 ===")
for label, arr in [('treat1', cace_t1_boot), ('treat2', cace_t2_boot)]:
    print(f"\n  [{label}]")
    print(f"    Bootstrap 평균   : {arr.mean():.4f}")
    print(f"    Bootstrap 표준편차: {arr.std():.4f}")
    print(f"    Bootstrap 95% CI : [{np.percentile(arr, 2.5):.4f}, "
          f"{np.percentile(arr, 97.5):.4f}]")
print("\n=== 단측 t-검정 결과 (귀무가설: CACE = 0) ===")
# treat1 결과
print(f"\n  [treat1] 우측 단측 검정 (H1: CACE > 0)")
print(f"    t 통계량  : {t1_stat:.4f}")
print(f"    p-value   : {t1_p:.4f}")
result_t1 = "귀무가설 기각 → treat1 유의한 BPday 증가 효과" if t1_p < ALPHA \
            else "귀무가설 기각 실패 → 유의하지 않음"
print(f"    결론      : {result_t1}")
# treat2 결과
print(f"\n  [treat2] 좌측 단측 검정 (H1: CACE < 0)")
print(f"    t 통계량  : {t2_stat:.4f}")
print(f"    p-value   : {t2_p:.4f}")
result_t2 = "귀무가설 기각 → treat2 유의한 BPday 감소 효과" if t2_p < ALPHA \
            else "귀무가설 기각 실패 → 유의하지 않음"
print(f"    결론      : {result_t2}")