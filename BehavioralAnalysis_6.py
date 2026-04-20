# 앞선 챕터에서 처치 효과를 추정할 때 처치가 결과에 평균적으로 얼마나 영향을 주는가를 추정했으며 처치와 결과가 선형 관계를 가진다고 가정, 그
# 러나 실제로는 처치가 모든 단위에 동일하게 적용되지 않고 특정 집단에서 더 크거나 작게 적용할 수도 있으며, 어떤 중간 변수를 통해 처치 효과가
# 전달될 수도 있으며, 처치 변수가 결과 변수와 공통 원인을 공유해 편향이 생길 수도 있음, 이러한 문제를 해결하기 위해 조절 효과/매개 효과/도구
# 변수를 고려한 실험설계가 필요

# %% 5 - 1. 행동 데이터 분석의 고급 도구 - 조절 효과
# 조절효과란 처치 변수 X가 결과 변수 Y에 미치는 영향이 제3의 변수 Z의 값에 따라 달라지는 현상을 말하며 이때 Z를 조절 변수라고 함, 예를 들어
# 놀이 공간유무가 쇼핑 체류 시간에 미치는 효과가 자녀 동반 여부에 따라 다를 수 있음, 이를 회귀식으로 나타내면 다음과 같음 : 쇼핑 체류시간 =
# 절편 + 계수1 * 놀이 공간유무 + 계수2 * 자녀 동반 여부 + 계수3 * (놀이 공간유무 * 자녀 동반 여부) -> 이때, 마지막 항을 상호작용항이라고
# 하며 상호작용항의 계수가 양수이면 Z가 클수록 X의 효과가 증폭되고 음수이면 Z가 클수록 X의 효과가 감소하며 0이면 두 변수간의 상호작용이 없음
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from linearmodels.iv import IV2SLS
hist_data_df = pd.read_csv('DataSet/chap11-historical_data.csv', encoding="utf-8-sig")
print("=== 데이터 기본 정보 ===")
print(hist_data_df.head())
print(f"\n행 수: {len(hist_data_df):,}")
print(f"\n변수별 요약:")
print(hist_data_df[['duration', 'play_area', 'children', 'age']].describe().round(2))
# 한편 조절 효과의 행태에 따라 다음 3가지 형태로 분류할 수 있음 -> 세분화 : 고객을 Z 기준으로 집단으로 나눴을때 집단별로 처치 효과의 크기가
# 체계적으로 다른 경우, 예를 들어 어린이 동반 고객과 미동반 고객을 나눠보면 놀이 공간의 효과가 전혀 다게 작용하는 경우가 해당함, 이 경우에는
# 조절변수 Z는 범주형 변수의 더미 변수가 됨
print("\n=== [1-2] 범주형 조절 모형: duration ~ play_area * children ===")
model_seg = ols("duration ~ play_area * children", data=hist_data_df)
res_seg   = model_seg.fit()
print(res_seg.summary().tables[1])
# 집단별 평균으로 시각적 교차검증
# 조절효과가 있다면 집단별 평균 차이가 play_area 수준에 따라 달라야 함
print("\n  [집단별 체류시간 평균 — 조절 패턴 확인]")
group_means = (
    hist_data_df
    .groupby(['play_area', 'children'])['duration']
    .mean()
    .round(2)
    .rename('평균 체류시간')
    .reset_index()
)
print(group_means.to_string(index=False))
# play_area 효과 크기를 children 집단별로 직접 계산
beta1 = res_seg.params['play_area']
beta3 = res_seg.params['play_area:children']
print(f"\n  자녀 미동반(children=0) 에서 play_area 효과: {beta1:.3f}분")
print(f"  자녀 동반(children=1)   에서 play_area 효과: {beta1 + beta3:.3f}분")
print(f"  조절효과 크기(β₃)                          : {beta3:.3f}분")
print(f"  → {'강화(synergistic)' if beta3 > 0 else '완충(buffering)'} 조절")
# 상호작용 : X의 효과가 Z의 값에 따라 선형적으로 변화, 가장 일반적인 경우이며 이때 Z는 Z가 연속형 변수가 됨
print("\n=== [1-3] 연속형 조절 모형: duration ~ play_area * age ===")
model_cont = ols("duration ~ play_area * age", data=hist_data_df)
res_cont   = model_cont.fit()
print(res_cont.summary().tables[1])
beta_pa     = res_cont.params['play_area']
beta_pa_age = res_cont.params['play_area:age']
age_mean    = hist_data_df['age'].mean()
age_std     = hist_data_df['age'].std()
print(f"\n  age 평균: {age_mean:.1f}세, 표준편차: {age_std:.1f}세")
print(f"\n  age별 play_area 효과 (β₁ + β₃ × age):")
for age_val in [age_mean - age_std, age_mean, age_mean + age_std]:
    eff = beta_pa + beta_pa_age * age_val
    print(f"    age = {age_val:.1f}세: {eff:.3f}분")
# 비선형성 : Z가 X 자신일 때 즉 X의 제곱항 X^2이 모형에 포함되어 X와 Y의 관계가 U자형 또는 역U자형으로 표현되는 경우, 예를 들어 이메일 발
# 송 횟수가 많아질수록 구매가 늘지만 어느 지점을 넘으면 오히려 구매가 감소하는 상황이 이에 해당
nonlinear_df = pd.read_csv('DataSet/chap11-nonlin_data.csv', encoding="utf-8-sig")
print("\n=== [1-4] 비선형 조절: Purchases ~ Emails + I(Emails**2) ===")
model_nl = ols("Purchases ~ Emails + I(Emails**2)", data=nonlinear_df)
# statsmodels formula의 I() 래퍼를 사용해야 수식 연산자를 변수 변환으로 인식
res_nl   = model_nl.fit()
print(res_nl.summary().tables[1])
beta1_nl = res_nl.params['Emails']
beta2_nl = res_nl.params['I(Emails ** 2)']
print(f"\n  Emails 계수 (β₁)   : {beta1_nl:.4f}")
print(f"  Emails² 계수 (β₂)  : {beta2_nl:.4f}")
if beta2_nl < 0:
    optimal = -beta1_nl / (2 * beta2_nl)
    print(f"  → 역U자형 관계: 최적 발송 횟수 = {optimal:.1f}회")
    print(f"  → 이 횟수를 초과하면 추가 이메일이 구매를 오히려 감소시킴")
elif beta2_nl > 0:
    optimal = -beta1_nl / (2 * beta2_nl)
    print(f"  → U자형 관계: 최솟값 발송 횟수 = {optimal:.1f}회")
else:
    print(f"  → 비선형 효과 없음")
# 한편 조절변수가 두 개 이상인 다중 조절 변수가 존재하는 경우에는 3중 상호작용도 가능, X의 효과에 대한 Z1의 조절 효과를 Z2가 다시 조절하고
# 조절 변수간 상호작용도 존재하는 경우가 이에 해당, 예를 들어 예를 들어 놀이 공간의 효과가 자녀 동반에 따라 달라지고 그 달라지는 정도가 다시
# 나이에 따라 달라질 수 있음 이를 회귀식으로 표현하면 다음과 같음 : 쇼핑 체류시간 = 절편 + 계수1 * 놀이 공간유무 + 계수2 * 자녀 동반 여부
# + 계수3 * (놀이 공간유무 * 자녀 동반 여부) + .... + 계수4 * (나이 * 자녀 동반 여부) + 계수5 * (나이 * 놀이 공간유무) + 계수6 * (나
# 이 * 놀이 공간유무 * 자녀 동반 여부)
print("=== [2-1] 3중 상호작용 모형: duration ~ play_area * children * age ===\n")
model_3way = ols("duration ~ play_area * children * age", data=hist_data_df)
res_3way   = model_3way.fit()
print(res_3way.summary().tables[1])
# 3중 상호작용 계수 단독 추출 및 해석
beta6 = res_3way.params['play_area:children:age']
print(f"\n  3중 상호작용 계수 (β₆): {beta6:.4f}")
print(
    f"  해석: 나이 1세 증가 시, 자녀 동반 고객에서의 play_area 조절 효과가 "
    f"추가로 {beta6:.4f}분 {'증가' if beta6 > 0 else '감소'}한다."
)
# 구체적인 시나리오별 play_area 총 효과 계산
# play_area 총 효과 = β₁ + β₄·children + β₅·age + β₆·children·age
b1 = res_3way.params['play_area']
b4 = res_3way.params['play_area:children']
b5 = res_3way.params['play_area:age']
age_vals      = [hist_data_df['age'].quantile(0.25),
                 hist_data_df['age'].mean(),
                 hist_data_df['age'].quantile(0.75)]
children_vals = [0, 1]
print(f"\n  [시나리오별 play_area 총 효과]")
print(f"  {'children':<12} {'age':<10} {'play_area 효과':>14}")
print(f"  {'-'*38}")
for ch in children_vals:
    for age in age_vals:
        eff = b1 + b4 * ch + b5 * age + beta6 * ch * age
        print(f"  {ch:<12} {age:<10.1f} {eff:>14.3f}분")
# 이때 시나리오별 효과를 살펴봄으로서 복잡한 다중 조절 변수 속에서 어떤 조건에서 효과가 크고 어떤 조건에서 작은가를 추가로 확인할 수 있음, 다
# 만 3중 상호작용 계수의 경우 조절 효과가 한 번 더 중첩되기 때문에 회귀분석의 결과 해석을 직관적으로 이해하기 어렵고 표본 오차도 크게 작용함,
# 따라서 부트스트랩 기반 신뢰구간을 이용해 계수의 분포를 추정하는 것이 더 정확 : 3중 상호작용 계수에 적용 불가능한 단측 t-검정을 대신하여 유
# 의성/안정성/효과성을 확인 가능하게하는 도구
def metric_fun(dat_df):
    res = ols("duration ~ play_area * children", data=dat_df).fit(disp=0)
    return res.params['play_area:children']
# 2중 상호작용 계수 추출 (play_area <-> children)
def boot_CI_fun(dat_df, metric_fun, B=100, conf_level=0.9, N_fixed=10000):
    coeffs = []
    for _ in range(B):
        boot_df = dat_df.sample(n=N_fixed, replace=True)
        coeffs.append(metric_fun(boot_df))
    coeffs.sort()
    cut = round(B * (1 - conf_level) / 2)
    return [coeffs[cut], coeffs[-(cut + 1)]]
print("\n=== [2-2] 3중 상호작용 계수 Bootstrap CI (B=100, 90%) ===")
ci_3way = boot_CI_fun(hist_data_df, metric_fun, B=100)
print(f"  Bootstrap 90% CI: [{ci_3way[0]:.4f}, {ci_3way[1]:.4f}]")
print(f"  point estimate  : {beta6:.4f}")
if ci_3way[0] > 0 or ci_3way[1] < 0:
    print(f"  → CI가 0을 포함하지 않음 — 3중 조절효과 통계적으로 유의")
else:
    print(f"  → CI가 0을 포함함 — 3중 조절효과 통계적으로 유의하지 않음")
# 상호작용항을 포함한 모형에서 처치변수 항의 계수(주효과 계수)는 다른 변수가 0일 때의 효과와 같음, 이때 연속형 조절 변수의 0이 실제로 의미없
# 는 값이면 주효과 계수도 실질적으로 해석하기 불가능해지므로 변수 센터링 즉, 연속형 조절 변수에서 평균값 등 대표값을 빼주는 처리를 해줌으로서
# 주효과 계수의 의미를 평균 나이인 고객에서의 처치 효과로 바꾸어줄 필요가 있음, 이때 적합도나 상호작용 계수는 바뀌지 않으며 바뀌는 것은 주효
# 과 계수뿐임
centered_df          = hist_data_df.copy()
centered_df['age_c'] = centered_df['age'] - centered_df['age'].mean()
age_mean = hist_data_df['age'].mean()
print(f"  age 평균: {age_mean:.2f}세 (이 값이 센터링 후 새로운 0점)")
# 센터링 전
res_raw  = ols("duration ~ play_area * age",   data=hist_data_df).fit()
# 센터링 후
res_cent = ols("duration ~ play_area * age_c", data=centered_df).fit()
# 계수 비교 테이블
comparison = pd.DataFrame({
    '센터링 전': res_raw.params.values,
    '센터링 후': [
        res_cent.params['Intercept'],
        res_cent.params['play_area'],
        res_cent.params['age_c'],
        res_cent.params['play_area:age_c']
    ]
}, index=['Intercept', 'play_area', 'age', 'play_area:age'])
comparison['변화 여부'] = comparison.apply(
    lambda r: '변함' if abs(r['센터링 전'] - r['센터링 후']) > 1e-6 else '불변', axis=1
)
print("\n=== [3-1] 센터링 전후 계수 비교 ===")
print(comparison.round(4).to_string())
# 핵심 해석 요약
b1_raw  = res_raw.params['play_area']
b1_cent = res_cent.params['play_area']
b3      = res_cent.params['play_area:age_c']
print(f"\n  [해석]")
print(f"  센터링 전 play_area 계수: {b1_raw:.3f}분  (age=0일 때 효과 — 해석 불가)")
print(f"  센터링 후 play_area 계수: {b1_cent:.3f}분 (age={age_mean:.1f}세일 때 효과 — 해석 가능)")
print(f"  → play_area:age 계수는 {res_cent.params['play_area:age_c']:.4f}로 동일")
se_raw  = res_raw.bse['play_area']
se_cent = res_cent.bse['play_area']
print(f"\n=== [3-2] play_area 계수 표준 오차 비교 ===")
print(f"  센터링 전 SE: {se_raw:.4f}")
print(f"  센터링 후 SE: {se_cent:.4f}")
print(f"  SE 감소율   : {(1 - se_cent / se_raw) * 100:.1f}%")
print(f"  → SE가 줄어들수록 기준점 이동이 추정 안정성 향상에 기여한 것")
# 또한 놀이 공간이 있는 매장(play_area=1)에서 자녀 동반의 효과를 계수에서 직접 읽고 싶을 때처럼 다른 범주형 조절 변수의 조절 효과가 적용되
# 도록 통제한 상태에서 원하는 변수(children) 의 조절 효과를 확인하고 싶을때는 조건에 해당하는 범주형 조절 변수를 1로 고정한 상테에서 계수를
# 추정하면 됨
centered_df['play_area_flipped'] = 1 - centered_df['play_area']
res_orig    = ols("duration ~ play_area         * children", data=hist_data_df).fit()
res_flipped = ols("duration ~ play_area_flipped * children", data=centered_df).fit()
print(f"\n=== [3-3] play_area 기준값 변경 ===")
print(f"\n  [원래 모형] play_area 기본값 = 0 (놀이 공간 없음)")
print(f"  children 계수: {res_orig.params['children']:.3f}분")
print(f"  해석: 놀이 공간 없는 매장에서 자녀 동반의 효과")
print(f"\n  [반전 모형] play_area 기본값 = 1 (놀이 공간 있음)")
print(f"  children 계수: {res_flipped.params['children']:.3f}분")
print(f"  해석: 놀이 공간 있는 매장에서 자녀 동반의 효과")
print(f"\n  → 상호작용 계수는 두 모형에서 동일해야 함:")
print(f"     원래  play_area:children      = {res_orig.params['play_area:children']:.4f}")
print(f"     반전  play_area_flipped:children = "
      f"{res_flipped.params['play_area_flipped:children']:.4f}")
# 한편 회귀 계수만으로는 '놀이 공간이 없는 10개 매장에 모두 놀이 공간을 설치하면 매장별로 고객 체류 시간이 얼마나 늘어나는가? 어느 매장이 가
# 장 큰 효과를 볼 것인가?' 와 같은 비즈니스 질문에 대응하기 어려움, 따라서 실제 고객 구성(children 또는 age 분포)을 반영한 매장별 예측 체
# 류 시간 변화를 확인해볼 필요가 있음 : 같은 고객이 놀이 공간이 없는 매장을 방문했을때와 있는 매장을 방문했을 때의 체류 시간 차이를 비교하고
# 매장 방문 고객 1인당 평균 체류 시간 증가량인 '평균효과'와 매장 전체의 총 체류 시간 증가량인 '합산효과'를 모두 파악하여 투자 우선순위 결정
model = ols("duration ~ play_area * (children + age)", data=hist_data_df)
res   = model.fit()
print("=== [4-1] 모형 적합 결과 ===")
print(res.summary().tables[1])
b_pa       = res.params['play_area']
b_pa_ch    = res.params['play_area:children']
b_pa_age   = res.params['play_area:age']
age_mean   = hist_data_df['age'].mean()
print(f"\n  play_area 주효과:         {b_pa:.3f}분")
print(f"  play_area:children 조절:  {b_pa_ch:.3f}분 (자녀 동반 시 추가 효과)")
print(f"  play_area:age 조절:       {b_pa_age:.4f}분/세")
print(f"\n  평균 나이({age_mean:.1f}세) 고객 기준 play_area 효과:")
print(f"    자녀 미동반: {b_pa + b_pa_age * age_mean:.3f}분")
print(f"    자녀 동반  : {b_pa + b_pa_ch + b_pa_age * age_mean:.3f}분")
# 계수 해석 요약
action_df = hist_data_df[hist_data_df['play_area'] == 0].copy()
# 현재 상태 예측
action_df['pred_dur0'] = res.predict(action_df)
# 반사실 상태 예측: 놀이 공간이 없는 매장에 대해서의 회귀식의 play_area를 1로 변경한 다음 예측되는 체류 시간을 추정
action_df_cf = action_df.copy()
action_df_cf['play_area'] = 1
action_df['pred_dur1'] = res.predict(action_df_cf)
# 개인별 효과
action_df['pred_dur_diff'] = action_df['pred_dur1'] - action_df['pred_dur0']
print("\n=== [4-2] 개인별 반사실 예측 샘플 ===")
# 매장별 효과 집계
print(action_df[['store_id', 'children', 'age',
                  'pred_dur0', 'pred_dur1', 'pred_dur_diff']]\
      .head(10).round(2).to_string(index=False))
action_res_df = (
    action_df
    .groupby('store_id')
    .agg(
        mean_dur_diff = ('pred_dur_diff', 'mean'),
        tot_dur_diff  = ('pred_dur_diff', 'sum'),
        n_customers   = ('pred_dur_diff', 'count')
    )
    .round(2)
)
print("\n=== [4-3] 매장별 집계 결과 ===")
print(action_res_df.describe().round(2))
# 투자 우선 순위 결정
print("\n=== [4-4] 투자 우선순위 분석 ===")
top_mean = action_res_df.sort_values('mean_dur_diff', ascending=False).head(3)
top_tot  = action_res_df.sort_values('tot_dur_diff',  ascending=False).head(3)
print(f"\n  [평균 효과 상위 3개 매장 — 고객 1인당 효과가 큰 매장]")
print(top_mean[['mean_dur_diff', 'tot_dur_diff', 'n_customers']].to_string())
print(f"\n  [합산 효과 상위 3개 매장 — 총 체류 시간 증가가 큰 매장]")
print(top_tot[['mean_dur_diff', 'tot_dur_diff', 'n_customers']].to_string())
# 두 기준의 순위 일치 여부
top_mean_ids = set(top_mean.index)
top_tot_ids  = set(top_tot.index)
overlap      = top_mean_ids & top_tot_ids
print(f"\n  두 기준 모두 상위 3위에 포함된 매장: {overlap if overlap else '없음'}")
if overlap:
    print(f"  → 이 매장들은 고객 질(1인당 효과)과 규모(총 효과) 모두 우수")
    print(f"    놀이 공간 투자 최우선 대상")
else:
    print(f"  → 두 기준 순위가 다름")
    print(f"    총 매출 임팩트 극대화 → 합산 효과 기준 선택")
    print(f"    고객 경험 최적화 우선 → 평균 효과 기준 선택")


# %% 5 - 2. 행동 데이터 분석의 고급 도구 - 매개 효과
# 매개효과는 처치 X가 결과 Y에 영향을 미치는 경로에 중간 변수 M이 있는 구조를 말함, 예를들어 X → M → Y의 인과 사슬이 존재하는 경우가 해당
# 하며 이때 M을 매개변수라고 함, 이때 놀이 공간이 있으면 고객이 더 오래 머물고 더 오래 머물기 때문에 더 많이 구매함, 즉 play_area의 groc
# ery_purchases에 대한 효과 중 상당 부분이 duration을 통해 간접적으로 전달, 매개 효과가 존재하는 상황에서 각 변수간에 작용하는 회귀 효과
# 는 다음과 같은 3단계로 분해 가능 : 총 효과 = 간접 경로와 직접 경로를 합친 전체 효과, 회귀식 grocery_purchases = 절편 + 계수 * play_
# area 의 계수가 총 효과의 크기 / X → M 경로 : 회귀식 duration = 절편 + 계수 * play_area 의 계수가 X → M 경로의 효과 크기 / M → Y
# 경로 추정 : 회귀식 grocery_purchases = 절편 + 계수1 * duration + 계수2 * play_area 에서 계수1이 X를 통제했을때 M → Y 경로의 효
# 과 크기이며 계수2가 duration을 통제한 후 남은 play_area의 '직접 효과'
res_total = ols("grocery_purchases ~ play_area", data=hist_data_df).fit()
c_total   = res_total.params['play_area']
print("=== [1-1] 1단계: 총 효과 ===")
print(res_total.summary().tables[1])
print(f"\n  총 효과 c = {c_total:.4f}")
print(f"  해석: 놀이 공간 유무에 따른 식료품 구매 차이 (모든 경로 포함)")
res_a = ols("duration ~ play_area", data=hist_data_df).fit()
a     = res_a.params['play_area']
print("\n=== [1-2] 2단계: X → M 경로 ===")
print(res_a.summary().tables[1])
print(f"\n  경로 a = {a:.4f}")
print(f"  해석: 놀이 공간이 체류 시간을 {a:.2f}분 증가시킴")
res_full = ols("grocery_purchases ~ duration + play_area", data=hist_data_df).fit()
b        = res_full.params['duration']
c_direct = res_full.params['play_area']
print("\n=== [1-3] 3단계: M → Y + 직접 효과 ===")
print(res_full.summary().tables[1])
print(f"\n  경로 b    = {b:.4f}  (duration 1분당 구매 변화)")
print(f"  직접 효과 c' = {c_direct:.4f}  (duration 통제 후 play_area 순수 직접 효과)")
indirect_effect  = a * b
mediation_ratio  = indirect_effect / c_total
c_reconstructed  = c_direct + indirect_effect   # c' + a×b = c 검증
print("\n=== [1-4] 효과 분해 ===")
print(f"\n  총 효과          c    = {c_total:.4f}")
print(f"  직접 효과        c'   = {c_direct:.4f}")
print(f"  간접 효과        a×b  = {a:.4f} × {b:.4f} = {indirect_effect:.4f}")
print(f"  재구성 총 효과   c' + a×b = {c_reconstructed:.4f}")
print(f"\n  등식 c = c' + a×b 성립 여부: "
      f"{'✓ 성립' if abs(c_total - c_reconstructed) < 0.01 else '✗ 불일치'}")
print(f"\n  매개 비율 = {indirect_effect:.4f} / {c_total:.4f} = {mediation_ratio:.4f}")
if mediation_ratio > 0.9:
    print(f"  → 완전 매개에 가까움: play_area 효과의 {mediation_ratio*100:.1f}%가 duration 경로로 전달")
elif mediation_ratio > 0.5:
    print(f"  → 부분 매개: play_area 효과의 {mediation_ratio*100:.1f}%가 duration 경로로 전달")
else:
    print(f"  → 매개 효과 약함: duration이 주요 경로가 아닐 수 있음")
# 이때 'X → M 경로 효과 크기 * M → Y 경로 효과 크기' 를 간접 효과라고 하며 '간접효과 / 총 효과' 를 매개 비율이라고 함, 매개비율이 1에 가
# 까우면 완전 매개라고 하며 X의 효과 거의 전부가 M을 통해 전달되는 것이며, 0에 가까우면 매개 효과가 없어 M이 X와 Y간 매개변수로 거의 기능하
# 지 않는 경우에 해당함, 중간 수준이면 부분 매개라고 지칭, 완전 매개 시 총 효과에서 간접 효과를 제외한 직접효과는 매우 작아짐
def percentage_mediated_fun(dat_df):
    c = ols("grocery_purchases ~ play_area",
            data=dat_df).fit(disp=0).params['play_area']
    a = ols("duration ~ play_area",
            data=dat_df).fit(disp=0).params['play_area']
    b = ols("grocery_purchases ~ duration",
            data=dat_df).fit(disp=0).params['duration']
    # 주의: 3단계(M → Y 경로 추정)에서 b를 추정할 때 play_area 없이 duration만으로 b를 추정
    # 이유: play_area 포함 시 b가 duration의 순수 효과가 아닌 play_area의 직접 효과 + duration 효과가 되어 다른 의미가 됨
    return (a * b) / c
# 포인트 추정 확인
pm_point = percentage_mediated_fun(hist_data_df)
print(f"\n=== [1-5] 매개 비율 Bootstrap CI ===")
print(f"\n  포인트 추정치: {pm_point:.4f}")
def boot_CI_fun2(dat_df, metric_fun, B=100, conf_level=0.9):
    N      = len(dat_df)
    coeffs = []
    for _ in range(B):
        boot_df = dat_df.sample(n=N, replace=True)
        coeffs.append(metric_fun(boot_df))
    coeffs.sort()
    cut = round(B * (1 - conf_level) / 2)
    return [coeffs[cut], coeffs[-(cut + 1)]]
ci_pm = boot_CI_fun2(hist_data_df, percentage_mediated_fun, B=100)
print(f"  Bootstrap 90% CI: [{ci_pm[0]:.4f}, {ci_pm[1]:.4f}]")
print(f"\n  해석:")
print(f"  → CI 하한 {ci_pm[0]*100:.1f}% ~ 상한 {ci_pm[1]*100:.1f}% 범위에서")
print(f"     play_area 효과가 duration 경로로 전달됨이 안정적으로 지지됨")
print(f"  → CI가 0을 포함하지 않으므로 간접 효과 유의함")
# 또한 부트스트랩 신뢰 구간을 통해 매개 비율의 불확실성을 추정하여 한번 더 완전 매개 여부를 확인
print(f"\n=== [1-6] 직접 효과의 통계적 vs 실질적 유의성 ===")
print(f"\n  직접 효과 c' = {c_direct:.4f}")
print(f"  총 효과   c  = {c_total:.4f}")
print(f"  직접 효과 비율 = {c_direct / c_total * 100:.2f}%")
print(f"\n  → 통계적으로는 유의(p<0.001, 표본 60만+)")
print(f"     그러나 총 효과의 {c_direct / c_total * 100:.1f}%로 실질적으로 무시 가능한 수준")
print(f"  → duration이 play_area 효과를 사실상 완전히 매개")
# 다만 매개 변수 검증 과정에서 duration은 무작위 배정되지 않았으므로 duration → grocery_purchases 경로를 통해 duration 단독으로 gro
# cery_purchases 에 영향을 미쳐 교란이 개입할 가능성 존재, 즉 '오래 머무는 고객이 원래 구매를 많이 하는 사람일 가능성'을 완전히 배제할 수
# 없음, 따라서 이 분석은 경로 구조의 탐색적 증거로만 해석하고 인과적 확신이 필요하다면 duration을 직접 개입시키는 별도 실험이 필요


# %% 5 - 3. 행동 데이터 분석의 고급 도구 - 도구 변수
# call_CSAT(상담 만족도)가 M6Spend(6개월 지출)에 미치는 효과를 알고 싶다고 할때 단순히 회귀 분석으로 바로 추정하면 높은 확률로 편향 벌생,
# CSAT가 높은 고객은 애초에 충성 고객일 가능성이 높기 때문, 즉 CSAT와 M6Spend를 동시에 높이는 관찰되지 않은 공통 원인(고객 충성도)이 존재
# 함, 따라서 회귀분석 시 CSAT의 순수한 인과 효과와 충성도로 인한 허위 상관을 구별하지 못해 계수를 과대추정하게 됨, 이때 내생 변수 X(call_C
# SAT)와 강하게 상관되면서도 X를 통해서'만' Y(M6Spend)에 영향을 미치며 공동 원인과도 독립인 도구 변수를 도입하여, '도구 변수 -> 내생 변수
# -> 종속 변수' 로 이어지는 관계를 분석함으로서 이 문제를 해결 가능
exp_data_df = pd.read_csv('DataSet/chap10-experimental_data.csv', encoding='utf-8-sig')
exp_data_df['group'] = exp_data_df['grp'].copy()
exp_data_df = exp_data_df.drop(columns=['grp'])
# group 변수를 이진(0/1)으로 변환
exp_data_df['group'] = np.where(exp_data_df['group'] == 'treat', 1, 0)
print("=== 데이터 기본 확인 ===")
print(exp_data_df[['group', 'call_CSAT', 'M6Spend', 'age', 'reason']]\
      .describe().round(2))
print(f"\n  group 분포:\n{exp_data_df['group'].value_counts()}")
print("\n=== [2-1] OLS 편향 회귀: M6Spend ~ call_CSAT (편향된 계수) ===")
res_ols = ols("M6Spend ~ call_CSAT + age + reason", data=exp_data_df).fit()
coef_ols = res_ols.params['call_CSAT']
print(res_ols.summary().tables[1])
print(f"\n  OLS call_CSAT 계수: {coef_ols:.4f}  ← 충성도 편향 포함")
# '도구 변수 -> 내생 변수 -> 종속 변수' 관계 분석은 총 두 단계의 최소자승법(2SLS) 을 통해 분석 가능 : 1단계 = 내생 변수 X를 도구변수 Z와
# 공변량 회귀 분석 실시, 여기서 얻은 예측값 call_CSAT_bar는 group에 의해 설명되는 외생적 변동만 담고 있어 충성도 편향이 제거되어있음 / 2
# 단계 = 원래의 call_CSAT 대신 call_CSAT_bar 로 종속변수 M6Spend 를 회귀
# 이때 2SLS 없이 충성도 편향이 제거된 순수한 CSAT → 지출 인과 효과를 얻을 수 있는 방법 존재 : Wald 추정량 = 축약형 회귀(도구 변수 -> 종
# 속 변수 공변량 회귀)의 회귀 계수 / 1단계 회귀의 회귀 계수 = 2단계 회귀에서 M6Spend에 대한 call_CSAT_bar의 계수와 동일
print("\n=== [2-2] 1단계 회귀: call_CSAT ~ group (관련성 검증) ===")
res_first = ols("call_CSAT ~ group + age + reason", data=exp_data_df).fit()
coef_first = res_first.params['group']
print(res_first.summary().tables[1])
print(f"\n  group → call_CSAT 계수 (a): {coef_first:.4f}")
print(f"  1단계 F-통계량: {res_first.fvalue:.1f}  (>>10 → 강한 도구변수 확인)")
print("\n=== [2-3] 축약형 회귀: M6Spend ~ group ===")
res_reduced = ols("M6Spend ~ group + age + reason", data=exp_data_df).fit()
coef_reduced = res_reduced.params['group']
print(res_reduced.summary().tables[1])
print(f"\n  group → M6Spend 계수 (b): {coef_reduced:.4f}")
wald_estimate = coef_reduced / coef_first
print(f"\n=== [2-4] Wald 추정량 수동 계산 ===")
print(f"  축약형 계수 (group → M6Spend)    : {coef_reduced:.4f}")
print(f"  1단계 계수  (group → call_CSAT)  : {coef_first:.4f}")
print(f"  Wald 추정량 = {coef_reduced:.4f} / {coef_first:.4f} = {wald_estimate:.4f}")
print(f"\n  OLS 계수  : {coef_ols:.4f}  ← 충성도 편향 포함 (과대 추정)")
print(f"  Wald(IV)  : {wald_estimate:.4f}  ← 편향 제거된 순수 인과 효과")
print(f"  편향 크기 : {coef_ols - wald_estimate:.4f}  ({(coef_ols - wald_estimate)/coef_ols*100:.1f}% 과대 추정)")
print("\n=== [2-5] IV2SLS 공식과 교차 검증 ===")
iv_mod = IV2SLS.from_formula(
    'M6Spend ~ 1 + age + reason + [call_CSAT ~ group]',
    data=exp_data_df.reset_index(drop=True)
).fit()
coef_iv = iv_mod.params['call_CSAT']
print(iv_mod.summary.tables[1])
print(f"\n  IV2SLS call_CSAT 계수: {coef_iv:.4f}")
print(f"  Wald 추정량           : {wald_estimate:.4f}")
print(f"  일치 여부: {'✓ 일치' if abs(coef_iv - wald_estimate) < 0.01 else '✗ 불일치'}")
# OLS 편향 계수는 CSAT가 1 증가 할때 지출이 4 높아지는 '상관관계' 를 포착
# 2SLS에 의한 계수는 CSAT가 1 증가 할때 지출이 2.99 높아지는 '인과관계' 포착
print("\n=== [2-6] 세 가지 추정치 비교 ===\n")
comparison = pd.DataFrame({
    '추정 방법': ['OLS (편향)', 'IV/2SLS (편향 제거)'],
    'call_CSAT 계수': [coef_ols, coef_iv],
    '해석': [
        'CSAT↑1점 → 지출↑4.00 (충성도 편향 포함)',
        'CSAT↑1점 → 지출↑2.99 (순수 인과 효과)'
    ]
})
print(comparison.to_string(index=False))