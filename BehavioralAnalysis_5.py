# 4 - 3. 실험 설계와 분석 - 군집 무작위 배정과 계층적 모델링
# 앞선 실험에서 관찰 단위를 각각 실험군/대조군에 할당했음 이는 각 관찰 단위가 독립적이라고 가정했기 때문, 그러나 현실 데이터는 자연스럽게 계
# 층구조를 이루는 경우가 많음, 예를 들어 여러 개의 콜센터 밑에 여러 명의 상담사가 있고 한 상담사가 여러건의 상담을 처리하는 것과 같음, 이때
# 한 상담사가 처리하는 상담은 서로 독립이 아님, 상담사의 친절도에 따라 고객 만족도가 달라지며 상담 부서에 따라 상담 내용이 달라짐 즉 같은 군
# 집(상담사) 안의 관측값들 사이에는 공통된 원인에 의한 내적 상관이 존재, 이를 무시하고 실험을 진행하면 군집 효과가 만들어낸 차이인데 처치 효
# 과로 잘못 탐지할 가능성이 커지며 군집 단위에서 배정 해야하는 현실적 제약을 반영하지 못함, 예를 들어 어느 콜센터에 새로운 교율을 실시했을때
# 해당 콜센터 안에서 어떤 통화는 새 교육 방식을 적용하고 어떤 통화는 기존 방식으로 처리하는 것은 현실적으로 불가능
import pandas as pd
import numpy as np
import itertools
from scipy.stats import chi2 as chi2_dist
import statsmodels.formula.api as smf
from scipy.spatial import distance_matrix as scipy_dist_matrix
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
hist_data_df = pd.read_csv('DataSet/chap10-historical_data.csv', encoding="utf-8-sig")
exp_data_df  = pd.read_csv('DataSet/chap10-experimental_data.csv', encoding="utf-8-sig")
for df in [hist_data_df, exp_data_df]:
    df['center_ID'] = df['center_ID'] + 100 # numpy 인덱스와 중복되지 않도록 센터 아이디에 100 을 더해서 서로 구분
hist_data_df.drop(columns=['M6Spend'], inplace=True, errors='ignore')
exp_data_df.drop(columns=['M6Spend'],  inplace=True, errors='ignore') # 미사용 컬럼 삭제
exp_data_df['group'] = exp_data_df['grp'].copy()
exp_data_df = exp_data_df.drop(columns=['grp'])
# 따라서 이러한 경우 개별 관찰 단위가 아니라 군집 전체를 하나의 단위로 실험군/대조군에 배정하는 군집 무작위 배정을 통해 이러한 문제를 해결할
# 수 있음, 일반적으로 처치 자체가 개인이 아닌 군집 단위로 적용될 수 밖에 없거나 개인 단위로 적용이 가능하더라도 군집 내 관측값들 사이의 오염
# 위험이 있는 경우에는 군집 무작위 배정 사용, 예를들어 새로운 교육 프로그램을 한 콜센터 전체에 도입하면서 그 콜센터의 일부 상담사에게만 적용
# 할 수 없는 상황이 전자에 해당하고 같은 콜센터 내의 상담사간 정보 공유로 인해 개인 단위로 배정하면 실험군과 대조군이 섞이는 오염이 발생하는
# 상황이 후자에 해당

# %% 계층 구조 확인 + 유효 표본 크기 추정
# 군집이 얼마나 강하게 관측값을 지배하는가 즉 전체 분산 중에서 군집간 분산이 차지하는 비율을 ICC(급내 상관계수) 라고 함, ICC가 0에 가까우면
# 군집이 관측값에 거의 영향을 미치지 않는다는 뜻이고 1에 가까우면 같은 군집 안의 관측값들이 거의 동일하다는 뜻, ICC를 통해 실질적으로 의미가
# 있는 표본수인 유효 표본 크기를 추정 가능, 유효 표본 크기는 ICC와 군집 크기에 반비례, ICC가 높고 군집 크기가 크면 실제로 확보한 관측수 N에
# 비해 통계적으로 유효한 정보량이 훨씬 적어지기 때문, 예를 들어 같은 상담사가 처리한 통화 1000건은 서로 강하게 상관되어 있으므로 독립적 통화
# 1000건만큼의 정보를 담고 있지 않음, 또한 데이터의 불균형 군집 크기를 확인하여 동일 계층 내의 군집 별로 군집 크기가 얼마나 불균형한지 측정,
# Bootstrap 단계에서 군집 단위 복원추출 시 표본 크기 결정에 활용
print("=== hist_data_df 기본 정보 ===")
print(hist_data_df.head())
print(f"\n전체 행 수     : {len(hist_data_df):,}")
print(f"컬럼 목록      : {hist_data_df.columns.tolist()}")
print(f"결측값 여부    :\n{hist_data_df.isnull().sum()}") # 기본 데이터 정보 확인
n_centers = hist_data_df['center_ID'].nunique()
n_reps    = hist_data_df['rep_ID'].nunique()
n_calls   = len(hist_data_df)
print(f"\n=== 계층 구조 ===")
print(f"콜센터 수 (L3) : {n_centers}")
print(f"상담사 수 (L2) : {n_reps}")
print(f"통화 건수 (L1) : {n_calls:,}") # 기본 계층 구조 확인
center_summary = hist_data_df.groupby('center_ID').agg(
    n_reps  = ('rep_ID',    'nunique'),
    n_calls = ('call_CSAT', 'count'),
    avg_CSAT = ('call_CSAT', 'mean')
).round(3)
print(f"\n=== 센터별 요약 ===")
print(center_summary)
print(f"\n센터당 상담사 수 : min={center_summary.n_reps.min()}, "
      f"max={center_summary.n_reps.max()}, "
      f"mean={center_summary.n_reps.mean():.1f}")
print(f"센터당 통화 건수 : min={center_summary.n_calls.min():,}, "
      f"max={center_summary.n_calls.max():,}") # 센터별 상담사 수 및 통화 건수 확인
rep_call_counts = hist_data_df.groupby('rep_ID')['call_CSAT'].count()
print(f"\n=== 상담사별 통화 건수 ===")
print(f"최솟값 : {rep_call_counts.min():,}")
print(f"최댓값 : {rep_call_counts.max():,}")
print(f"평균   : {rep_call_counts.mean():.1f}")
print(f"중앙값 : {rep_call_counts.median():.1f}") # 상담사별 통화 건수 불균형 확인
null_model = smf.mixedlm("call_CSAT ~ 1",
                         data=hist_data_df,
                         groups=hist_data_df["center_ID"])
null_fit   = null_model.fit(reml=True)
var_between = float(null_fit.cov_re.iloc[0, 0])   # 센터 간 분산
var_within  = float(null_fit.scale)                # 잔차(센터 내) 분산
icc_center  = var_between / (var_between + var_within)
print(f"\n=== ICC 계산 (센터 수준) ===")
print(f"센터 간 분산 (σ²_between) : {var_between:.4f}")
print(f"잔차 분산    (σ²_within)  : {var_within:.4f}")
print(f"ICC                       : {icc_center:.4f}")
if icc_center >= 0.1:
    icc_verdict = "높음 → 군집 구조를 반드시 모형에 반영해야 함"
elif icc_center >= 0.05:
    icc_verdict = "중간 → 군집 구조 반영 권장"
else:
    icc_verdict = "낮음 → OLS로도 큰 문제 없을 수 있음"
print(f"해석                      : {icc_verdict}") # 센터 수준 ICC 계산 : 센터 간 분산 / (센터 간 분산 + 잔차 분산)
n_j   = center_summary['n_calls'].mean()
N     = n_calls
N_eff = N / (1 + icc_center * (n_j - 1))
print(f"\n=== 유효 표본 크기 ===")
print(f"실제 통화 건수 (N)     : {N:,}")
print(f"센터 평균 크기 (n_j)   : {n_j:,.1f}")
print(f"유효 표본 크기 (N_eff) : {N_eff:,.0f}")
print(f"정보 손실률            : {(1 - N_eff/N)*100:.1f}%") # 유효 표본 크기 계산 : N / (1 + ICC * (n_j - 1))

# %% 혼합 선형 모형
# 군집 구조가 있는 데이터를 올바르게 분석하기 위해서는 일반적인 OLS가 아닌 혼합 선형 모형(Mixed Linear Model 혹은 계층적 선형 모형)을 사
# 용해야함, 일반 OLS는 모든 계수를 고정 효과로 추정하지만 즉 모든 관찰 단위에 동일하게 적용되는 하나의 절편과 기울기를 가정하지만, 혼합 모형
# 은 여기에 랜덤 효과를 추가하여 각 군집이 전체 평균에서 얼마나 벗어나는지를 군집마다 별도로 추정하여 보정하기 때문, 예를 들면 고객 만족도 =
# 계수1 * 변수1 + 계수2 * 변수2 + .... + 전체 평균 절편 + 센터별 랜덤 절편(해당 센터가 평균에서 벗어나있는 정도) + 상담사별 랜덤 절편(해
# 당 상담사가 평균에서 벗어나있는 정도) 로 추정, 따라서 군집효과로 인한 편향과 노이즈 억제 가능
# 고정 효과는 모집단 전체에 적용되는 평균적인 관계를 의미하며 통화 이유/고객 나이 등의 변수가 고객 만족도에 미치는 영향은 각 센터/상담사마다
# 동일함, 반면 랜덤 효과는 특정 센터나 특정 상담사가 전체 평균에서 얼마나 벗어나는지를 나타내는 개별 편차이며 특정 센터가 전체 평균보다 만족
# 도가 높은 것은 그 센터의 특수한 사정때문인 것이지 모집단 전체에 적용할 수 있는 규칙이 아님, 랜덤 효과는 이런 개별 편차를 정규분포를 따르는
# 확률 변수로 모델링함
model1 = smf.mixedlm(
    "call_CSAT ~ reason + age", # 고정 효과
    data   = hist_data_df,
    groups = hist_data_df["center_ID"]   # 최상위 군집 변수 즉, 센터 수준 랜덤 절편 추가
)
fit1 = model1.fit(reml=True)
print("=== 모형 1: 센터 랜덤 절편만 포함 ===")
print(fit1.summary())
var_center_1 = float(fit1.cov_re.iloc[0, 0])
var_resid_1  = float(fit1.scale)
icc_1        = var_center_1 / (var_center_1 + var_resid_1)
print(f"\n  센터 간 분산 (σ²_center) : {var_center_1:.4f}")
print(f"  잔차 분산    (σ²_resid)  : {var_resid_1:.4f}")
print(f"  ICC (센터 수준)          : {icc_1:.4f}")
# 센터 수준 랜덤 효과만 포함
vcf = {"rep_ID": "0+C(rep_ID)"} # 키(rep_ID) : 분산 성분에 부여할 이름  # 값(0 + C(rep_ID)) : rep_ID를 범주형으로 처리하되 절편
# 없이 각 수준별 분산만 추정 => 각 상담사별 고유한 랜덤 절편을 부여
model2 = smf.mixedlm(
    "call_CSAT ~ reason + age",
    data        = hist_data_df,
    groups      = hist_data_df["center_ID"],
    re_formula  = '1',       # 센터 수준(최상위 군집 변수)에서는 랜덤 기울기 랜덤 기울기 미허용
    vc_formula  = vcf        # 추가 랜덤 효과 = 중간 군집 계층 즉, 상담사 수준 랜덤 절편 추가
)
fit2 = model2.fit(reml=True)
print("\n=== 모형 2: 센터 + 상담사 이중 랜덤 효과 ===")
print(fit2.summary())
var_center_2 = float(fit2.cov_re.iloc[0, 0])
var_rep_2    = float(fit2.vcomp[0])     # vc_formula로 추가한 상담사 분산
var_resid_2  = float(fit2.scale)
var_total_2  = var_center_2 + var_rep_2 + var_resid_2
icc_center_2 = var_center_2 / var_total_2
icc_rep_2    = var_rep_2    / var_total_2
icc_resid_2  = var_resid_2  / var_total_2
print(f"\n  센터 간 분산    (σ²_center) : {var_center_2:.4f}  ({icc_center_2*100:.1f}%)")
print(f"  상담사 간 분산  (σ²_rep)    : {var_rep_2:.4f}  ({icc_rep_2*100:.1f}%)")
print(f"  잔차 분산       (σ²_resid)  : {var_resid_2:.4f}  ({icc_resid_2*100:.1f}%)")
print(f"  전체 분산       (σ²_total)  : {var_total_2:.4f}")
# 센터 + 상담사 수준 이중 랜덤 효과 포함
ll1 = fit1.llf
ll2 = fit2.llf
lrt_stat  = -2 * (ll1 - ll2)
lrt_p     = chi2_dist.sf(lrt_stat, df=1)   # df=1 : 상담사 분산 파라미터 1개 추가
print(f"\n=== 모형 비교 (로그 우도 비율 검정) ===")
print(f"  모형 1 로그 우도 : {ll1:.2f}")
print(f"  모형 2 로그 우도 : {ll2:.2f}")
print(f"  LRT 통계량       : {lrt_stat:.2f}")
print(f"  p-value          : {lrt_p:.6f}")
print(f"  결론 : {'모형 2 채택 (상담사 랜덤 효과 유의)' if lrt_p < 0.05 else '모형 1로 충분'}")
# 각 모형 비교 : 로그 우도 검정을 통해 제약 모형(센터 랜덤 효과만 인정)에 비해 비제약 모형(센터 + 상담사 랜덤효과 인정)이 데이터를 얼마나 더
# 잘 설명하는 지 예측, P-value가 유의하면 제약 모형의 성능이 유의미하게 좋으며 상담사 랜덤 효과가 존재한다고 판단
re_df = pd.DataFrame({
    'center_ID'  : fit1.random_effects.keys(),
    'random_eff' : [v['Group'] for v in fit1.random_effects.values()]
}).sort_values('random_eff', ascending=False).reset_index(drop=True)
print(f"\n=== 센터별 랜덤 절편 추정치 (BLUP) ===")
print(re_df.to_string(index=False))
print(f"\n  전체 평균 고정 절편 : {fit1.fe_params['Intercept']:.4f}")
print(f"  랜덤 효과 범위      : [{re_df.random_eff.min():.4f}, "
      f"{re_df.random_eff.max():.4f}]")
# 각 센터의 랜덤 절편 추정치 확인

# %% 군집 단위 층화 배정
# 군집 단위 무작위 배정 시 유효 표본 수가 크게 줄어드므로 배정 단위 수가 매우 적어짐, 즉 그룹 간 공변량 불균형이 우연히 심하게 발생할 확률이
# 매우 높아지므로 페어링 기반 층화 배정으로 특성이 비슷한 2개 그룹씩 묶어 그룹의 각 군집을 실험군/대조군에 배정하여 공변량 불균형 해소
center_data_df = hist_data_df.groupby('center_ID').agg(
    nreps          = ('rep_ID',    lambda x: x.nunique()),
    avg_call_CSAT  = ('call_CSAT', 'mean'),
    avg_age        = ('age',       'mean'),
    pct_reason_pmt = ('reason',    lambda x: (x == 'payment').mean())
).round(4)
center_data_df['nreps'] = center_data_df['nreps'].astype(float) # MinMaxScaler 처리를 위해 nreps를 float으로 변환
print("=== 센터 수준 집계 데이터 ===")
print(center_data_df)
# 센터 수준 집계 데이터 생성 : 배정 단위가 센터이므로 처치 배정 전에 이미 결정되어있고 처치 효과 추정에 영향을 미칠 수 있는 공변량(고객 나이
# , 통화이유, 고객 만족도) 를 센터 단위로 집계
def strat_prep_fun(dat_df):
    num_df    = dat_df.loc[:, dat_df.dtypes == 'float64'].copy()
    center_ID = list(dat_df.index)   # 인덱스 = center_ID (101~110)
    scaler = MinMaxScaler()
    num_np = scaler.fit_transform(num_df)
    return center_ID, num_np
# 변수 정규화 : center_data_df의 변수들을 MinMaxScale 후 인덱스인 center_ID 반환
def pair_fun(dat_df, K=2):
    match_len = K - 1     # = 1 : 각 센터에 대해 가장 가까운 이웃 1개를 찾음
    match_idx = match_len - 1   # = 0 : argpartition의 kth 파라미터
    center_ID, data_np = strat_prep_fun(dat_df)
    N = len(data_np)
    # 거리 행렬 계산 및 대각선 처리
    d_mat = scipy_dist_matrix(data_np, data_np)
    np.fill_diagonal(d_mat, N + 1)   # 자기 자신 선택 방지
    available_temp = list(range(N))
    matches_lst    = []
    lim            = int(N / match_len)   # 최대 페어 수 상한
    # argpartition : 각 행에서 거리가 가장 작은 kth개의 인덱스를 앞으로 모음
    closest = np.argpartition(d_mat, kth=max(match_idx, 1), axis=1)
    for n in range(N):
        if len(matches_lst) == lim:
            break
        if n not in available_temp:
            continue
        # 탐색 범위를 점진적으로 넓혀가며 available_temp에서 짝을 찾음
        for match_lim in range(max(match_idx, 1), N - 1):
            possible_matches = closest[n, :match_lim].tolist()
            matches = list(set(available_temp) & set(possible_matches))
            if len(matches) == match_len:
                matches.append(n)
                matches_lst.append(matches)
                available_temp = [m for m in available_temp if m not in matches]
                break
            else:
                closest[n, :] = np.argpartition(d_mat[n, :], kth=match_lim)
    # numpy 인덱스를 실제 center_ID로 변환 : matches_lst의 각 원소는 [idx_a, idx_b] 형태의 numpy 행 인덱스, center_ID[k]로 실제
    # 센터 번호(101~110)로 매핑
    matches_id_lst = [
        [center_ID[pair[0]], center_ID[pair[1]]]
        for pair in matches_lst
    ]
    return np.array(matches_id_lst)
# 군집 간 페어링 : stratified_assgnt_fun과 매커니즘 동일, 다만 numpy 인덱스를 center_ID로 변환 필요
stratified_pairs = pair_fun(center_data_df, K=2)
print("\n=== 층화 배정 페어 결과 ===")
print(f"{'페어':>4s}  {'센터 A':>8s}  {'센터 B':>8s}")
print("-" * 28)
for i, pair in enumerate(stratified_pairs):
    print(f"  {i+1}     {int(pair[0]):>6d}     {int(pair[1]):>6d}")
# 페어링 실행 및 결과 확인
print("\n=== 페어 내 특성 차이 확인 ===")
check_vars = ['avg_call_CSAT', 'nreps', 'avg_age', 'pct_reason_pmt']
for i, pair in enumerate(stratified_pairs):
    a_vals = center_data_df.loc[pair[0], check_vars]
    b_vals = center_data_df.loc[pair[1], check_vars]
    diff   = (a_vals - b_vals).abs()
    print(f"\n  페어 {i+1} (센터 {int(pair[0])} vs {int(pair[1])})")
    for v in check_vars:
        print(f"    {v:>20s} | A={a_vals[v]:.4f}  B={b_vals[v]:.4f}  "
              f"차이={diff[v]:.4f}")
# 페어링 품질 확인

# %% 부트스트랩 기반 검정력 분석
# 실험군/대조군 배정은 콜센터별로 이루어졌으나 센터 단위 부트스트랩은 센터수가 너무 적어 불안정, 따라서 검정력 분석은 전체 상담사 담위로 실시
def hlm_metric_fun(dat_df):
    vcf   = {"rep_ID": "0+C(rep_ID)"}
    h_mod = smf.mixedlm(
        "call_CSAT ~ reason + age + group",
        data       = dat_df,
        groups     = dat_df["center_ID"],
        re_formula = '1',
        vc_formula = vcf
    )
    return h_mod.fit(reml=True, disp=False).fe_params['group[T.treat]']
# 혼합 선형 모형의 처치 계수 반환 헬퍼 함수 : 실험의 목표 변수를 혼합 선형 모형에서 처치(group)의 계수로 정의
def boot_CI_fun(dat_df, metric_fun=hlm_metric_fun,
                B=20, conf_level=0.9, Ncalls_rep=1200):
    rng        = np.random.default_rng()
    rep_ids    = dat_df['rep_ID'].unique()
    coeff_boot = []
    for _ in range(B):
        # 1) 상담사 단위 복원추출
        sampled_reps = rng.choice(rep_ids, size=len(rep_ids), replace=True)
        # 2) 뽑힌 상담사별 Ncalls_rep건씩 복원추출 + 중복 상담사 ID 구분
        frames = []
        for i, rep in enumerate(sampled_reps):
            grp = dat_df.loc[dat_df['rep_ID'] == rep].sample(
                n=Ncalls_rep, replace=True
            ).copy()
            grp['rep_ID'] = f"{rep}_{i}"
            frames.append(grp)
        boot_df = pd.concat(frames, ignore_index=True)
        coeff_boot.append(metric_fun(boot_df))
    coeff_boot.sort()
    offset = max(round(B * (1 - conf_level) / 2), 1)
    return [coeff_boot[offset], coeff_boot[-(offset + 1)]]
# 처치 계수의 부트스트랩 신뢰구간 도출 헬퍼 함수 : 중복을 제외한 상담사의 수만큼 상담사 복원추출 후 각 상담사에 대해서 Ncalls_rep건씩 복원
# 추출하여 표본을 만든 다음 해당 표본을 혼합 선형 모형에 적합시켜 처치 계수를 얻기를 B번 반복, B개 계수의 분포를 바탕으로 신뢰구간 추청
def decision_fun(dat_df, metric_fun=hlm_metric_fun,
                 B=20, conf_level=0.9, Ncalls_rep=1200):
    ci = boot_CI_fun(dat_df, metric_fun,
                     B=B, conf_level=conf_level, Ncalls_rep=Ncalls_rep)
    return 1 if ci[0] > 0 else 0
# 도출된 부트스트랩 신뢰구간 기반 단일 배정 의사결정 헬퍼 함수
# 각 상담사를 복원 추출하고 다시 각 상담사의 통화 내역을 '동일한 크기'로 복원추출한 표본으로 혼합 모형 적합 후 처치 계수 집계 : 통화 내역을
# 행 별로 추출 시 상담사 군집 구조가 무시되는 반면 상담사를 단위로 추출하면 상담사 랜덤 효과가 모델에 제대로 반영, 또 상담사간 총 통화건수가
# 다르므로 통화가 많은 상담사의 과도한 영향력을 억제하고 총 데이터 크기를 통일하기 위해 각 상담사로부터 동일한 크기의 통과 내역 추출 => 이때
# 반복 집계된 처치 계수의 부트스트랩 CI 하한이 0 초과이면 처치 효과 탐지 성공(1), 아니면 실패(0)
def perm_to_treat_centers(perm, stratified_pairs):
    Npairs  = len(stratified_pairs)
    bin_str = f'{perm:0{Npairs}b}'
    return [stratified_pairs[i][int(d)] for i, d in enumerate(bin_str)]
# 페어별 군집 배정 헬퍼 함수 : perm을 이진수로 변환, 이진수의 각 비트는 좌측부터 페어0, 페어1... 에 해당, 비트가 0이면 해당 페어의 첫번째
# 군집이 1이면 두번째 군집이 실험군에 배정, 예를 들어 perm=5, Npairs=5 일시 십진수 5를 이진수로 비트 수 5개짜리 이진수로 변환하면 00101
# 이때 페어0: 0 = 첫번째, 페어1: 0 = 첫번째, 페어2: 1 = 두번째, 페어3: 0 = 첫번째, 페어4: 1 = 두번째를 실험군에 배정됨
def power_sim_fun(dat_df, metric_fun=hlm_metric_fun,
                  Ncalls_rep=250, eff_size=1,
                  B=5, conf_level=0.9):
    stratified_pairs = pair_fun(center_data_df, K=2)
    Npairs = len(stratified_pairs)
    Nperm  = 2 ** Npairs     # = 32
    power_list = []
    # 모든 군집 페어링 후 모든 페어를 크기 2짜리(2비트) 페어의 5개 순열로 반환 = 총 2**5 = 32가지 군집 배정 경우의 수 발생
    for m in sorted(dat_df['month'].unique()): # 월별로 시뮬레이션 수행
        print(f"  {m} 월 처리 중... (순열 {Nperm}가지)")
        month_df = pd.concat(
            [grp.sample(n=Ncalls_rep, replace=True)
             for _, grp in dat_df.loc[dat_df['month'] == m].groupby('rep_ID', group_keys=False)],
            ignore_index=True
        )
        # 순열 순회 전 월별 데이터수 균일화, 월별로 각 상담사별 통화 내역수를 동일하게 통일, 그 이유 및 작동과정은 boot_CI_fun에서와 유사
        for perm in range(Nperm):
            print(f"  순열 {perm} 번째 처리 중...")
            # 가능한 모든 배정 순열 순회: 비군집화 배정에서는 표본이 많으므로 무작위로 실험하더라도 매번 다른 배정 조합이 나올 확률이 높음,
            # 그러나 군집화 배정 시 표본 수가 크게 줄어드므로 특정 조합만 많이 선택되고 특정 조합은 선택되지 않은 확률이 크게 증가하는 반면
            # 전수조사의 계산 비용은 감당할만 하므로, 5개 그룹의 각 군집에 처치가 할당되는 모든 32가지 경우를 순회
            treat_centers = perm_to_treat_centers(perm, stratified_pairs)
            # 이번 순회에 해당하는 페어별 처치 군집 선택
            sim_df         = month_df.copy()
            sim_df['group'] = np.where(
                sim_df['center_ID'].isin(treat_centers), 'treat', 'ctrl'
            )
            mask = sim_df['group'] == 'treat'
            sim_df.loc[mask, 'call_CSAT'] = np.clip(
                sim_df.loc[mask, 'call_CSAT'] + eff_size,
                a_min=None, a_max=10
            )
            # 실험군에 효과 적용 : 실험군에 eff_size 더하되 CSAT 최대 상한은 10으로 제한
            D = decision_fun(sim_df, metric_fun,
                             B=B, conf_level=conf_level,
                             Ncalls_rep=Ncalls_rep)
            power_list.append(D)
            # 처치 계수의 부트스트랩 신뢰구간 산출 후 탐지 여부 판단
    return power_list
# 군집 페어링 후 부스트스트랩 시뮬레이션 기반 검정력 분석 함수
Nperm    = 2 ** len(pair_fun(center_data_df, K=2))
print("=== 검정력 시뮬레이션 시작 ===")
print(f"  eff_size  : 1")
print(f"  B         : 5")
print(f"  conf_level: 0.9\n")
power_results = power_sim_fun(
    hist_data_df,
    metric_fun  = hlm_metric_fun,
    Ncalls_rep  = 250,
    eff_size    = 1,
    B           = 5,
    conf_level  = 0.9
)
print(f"\n=== 검정력 시뮬레이션 결과 ===")
print(f"  전체 시뮬레이션 수 : {len(power_results)}")
print(f"  탐지 성공 횟수     : {sum(power_results)}")
print(f"  추정 검정력        : {np.mean(power_results):.4f}  "
      f"({np.mean(power_results)*100:.1f}%)")
# 검정력 분석 결과 (탐색용, B = 5, Ncalls_rep = 250)

# %% 실험 결과 분석
print("=== 실험 데이터 기본 정보 ===")
print(f"  총 행 수         : {len(exp_data_df):,}")
print(f"  센터 수          : {exp_data_df['center_ID'].nunique()}")
print(f"  상담사 수        : {exp_data_df['rep_ID'].nunique()}")
print(f"  처치/대조 배분   :")
print(exp_data_df.groupby('group')['center_ID'].nunique()
      .rename('센터 수').to_frame().to_string())
print()
print(exp_data_df.groupby('group')['call_CSAT']
      .agg(['count', 'mean', 'std'])
      .rename(columns={'count':'통화 수', 'mean':'CSAT 평균', 'std':'CSAT 표준편차'})
      .round(4))
# 데이터 기본 확인
def balance_check(dat_df, covariates):
    treat = dat_df.loc[dat_df['group'] == 'treat']
    ctrl  = dat_df.loc[dat_df['group'] == 'ctrl']
    rows = []
    for col in covariates:
        if dat_df[col].dtype == 'object' or dat_df[col].nunique() <= 5:
            # 범주형: 빈도 비율 비교
            t_prop = treat[col].value_counts(normalize=True)
            c_prop = ctrl[col].value_counts(normalize=True)
            for cat in dat_df[col].unique():
                tp = t_prop.get(cat, 0)
                cp = c_prop.get(cat, 0)
                p_pool = (tp + cp) / 2
                denom  = np.sqrt(p_pool * (1 - p_pool)) if p_pool not in (0, 1) else np.nan
                smd    = (tp - cp) / denom if denom else np.nan
                rows.append({
                    '변수': f"{col}={cat}",
                    '처치 비율': round(tp, 4),
                    '대조 비율': round(cp, 4),
                    'SMD': round(smd, 4) if not np.isnan(smd) else np.nan
                })
        else:
            # 연속형: 평균 비교
            t_mean, t_std = treat[col].mean(), treat[col].std()
            c_mean, c_std = ctrl[col].mean(), ctrl[col].std()
            pool_std = np.sqrt((t_std**2 + c_std**2) / 2)
            smd = (t_mean - c_mean) / pool_std if pool_std > 0 else np.nan
            rows.append({
                '변수': col,
                '처치 평균': round(t_mean, 4),
                '대조 평균': round(c_mean, 4),
                'SMD': round(smd, 4)
            })
    return pd.DataFrame(rows)
print("\n=== 균형 검사 결과 ===")
balance_df = balance_check(exp_data_df, covariates=['reason', 'age'])
print(balance_df.to_string(index=False))
smd_vals = balance_df['SMD'].dropna().abs()
if (smd_vals > 0.2).any():
    print("\n  ⚠ |SMD| > 0.2 변수 존재 — 공변량 조정 모형 필수")
elif (smd_vals > 0.1).any():
    print("\n  △ |SMD| 0.1~0.2 변수 존재 — 공변량 조정 모형 권장")
else:
    print("\n  ✓ 모든 |SMD| < 0.1 — 균형 양호")
# 균형 검사 : 처치군/대조군의 나이, 통화 이유 등 변수의 표준화 평균 차이를 비교해 우연히 처치군/대조군의 공변량 분포가 달라졌는지를 확인
print("\n=== 혼합 선형 모형 결과 (Point Estimate) ===")
vcf   = {"rep_ID": "0+C(rep_ID)"}
h_mod = smf.mixedlm(
    "call_CSAT ~ reason + age + group",
    data       = exp_data_df,
    groups     = exp_data_df["center_ID"],
    re_formula = '1',
    vc_formula = vcf
)
hlm_result = h_mod.fit(reml=True, disp=False)
treat_coeff = hlm_result.fe_params['group[T.treat]']
treat_pval  = hlm_result.pvalues['group[T.treat]']
treat_se    = hlm_result.bse['group[T.treat]']
print(f"  처치 효과 계수  : {treat_coeff:.4f}")
print(f"  표준 오차       : {treat_se:.4f}")
print(f"  p-value         : {treat_pval:.4f}")
# 혼합 선형 모형 적합 : 처치 효과 계수 및 P-value를 확인, 정규 근사는 군집 수가 충분히 클 때만 Z-검정의 정규근사가 성립하므로 군집화 배정
# 에서는 P-value가 유의하지하지 않게 나올 수 있음 따라서 P-value는 단순 참고용
print("\n=== Bootstrap 신뢰구간 (B=5, conf_level=0.9) ===")
boot_ci = boot_CI_fun(
    exp_data_df,
    metric_fun = hlm_metric_fun,
    B          = 5,
    conf_level = 0.9,
    Ncalls_rep = 1200
)
print(f"  Bootstrap CI (90%) : [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]")
print(f"  처치 효과 계수     : {treat_coeff:.4f}")
# 부트스트랩 신뢰 구간 확인(탐색용, B = 5) : 부트스트랩 신뢰 구간의 하한 확인
print("\n=== 최종 분석 결과 ===")
if boot_ci[0] > 0:
    verdict = "처치 효과 통계적으로 유의 (CI 하한 > 0)"
    symbol  = "✓"
else:
    verdict = "처치 효과 통계적으로 유의하지 않음 (CI 하한 ≤ 0)"
    symbol  = "✗"
print(f"  {symbol} {verdict}")
print(f"\n  처치 효과 point estimate : {treat_coeff:.4f}")
print(f"  Bootstrap 90% CI         : [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]")
print(f"  해석: 새 교육 프로그램은 상담사당 평균 CSAT를 "
      f"{treat_coeff:.2f}점 {'올렸으며' if treat_coeff > 0 else '낮췄으며'},"
      f" 이 효과는 90% 신뢰수준에서 {'통계적으로 유의하다.' if boot_ci[0] > 0 else '통계적으로 유의하지 않다.'}")
# 최종 판단 및 결과 요약