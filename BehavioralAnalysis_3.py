# 4 - 1. 실험 설계와 분석 - 단순 무작위 배정 기반
# 실험이란 어떤 행동의 원인을 알아보기 위해 연구자가 조건을 의도적으로 바꾸고 그 결과를 관찰하는 것, 일반적으로 실험의 인과관계 다이어그램은
# 다음과 같음 : 개입 => 행동 논리 => 목표 지표 => 비즈니스 목표 : 먼저 비즈니스 목표와 목표지표를 정의한 후 개입/행동논리 순서로 정의
# 비즈니스 목표 : 개입을 통해 기업이 얻고자 하는 최종 효과, 수익/비용/고객 유지율과 같이 구체적이고 명확한 변수로 정의해야함
# 목표 지표 : 실험의 성공 여부 즉 비즈니스 목표 달성 여부를 측정하는 기준, 일반적으로 수익/비용에 직결된 지표를 사용하나 노이즈 감소를 위해
# 개입에 가까운 지표를 사용할 수도 있음, 이때 신뢰성있게 측정 가능한 지표를 선택해야하고('예약 난이도' 보다는 '고객의 예약 서비스 만족도'가
# 더 신뢰성있게 측정 가능) 실험 사전에 가능한 간결한 목표 지표를 선정해야함(실험 사후 선정 시 분석자의 편의에 맞게 지표를 취사선택할 위험성
# 이 증가하며 지나치게 복합적인 지표는 결과를 모호하게 만듦, 지표의 존재와 영향을 충분히 고려하여야만 복합 지표를 제한적으로 사용 가능)
# 개입 : 기업이 실험군에 취하는 조치, 개입 자체는 단순하더라도 개입의 여러 복합적인 요소(예를 들어 원클릭 예약 버튼이라면 버튼의 위치/색상/
# 크기/내용/이후 절차/필요한 정보 등)를 고려해야 하며, 개입이 목표 지표를 변동시키는 맥락도 고려해야함(예를 들어 예약 버튼이 목표 지표를 개
# 선 시켰을때 예약 과정에 걸리는 시간단축 자체가 매력적이어서인지 아니면 예약 메뉴의 접근성이 낮았기 때문인지 예약 메뉴의 특정 과정에 문제가
# 있어서 인지 등), 따라서 가능한 하나의 개입에 대해서 다양한 요소를 변동시켜가며 개입과 목표지표간의 관계를 탐색하되 실험하는 개입의 종류 자
# 체는 최소화해야함, 이때 여러 요소의 변동을 주었음에도 개입이 목표 지표를 일관되게 개선시킨다면 개입 자체가 효과성이 있는 것
# 행동 논리 : 정의한 개입이 목표 지표에 영향을 미치는 이유 및 과정, 행동 논리에서는 합리적이며 현실적인 행동의 전개를 논리적으로 표현 할 수
# 있어야하며, 이를 위해 최소한 일부 효과를 명확하게 관찰 가능하면서도 가능한 작은 구성요소 단위로 세분화된 내용으로 설정하는 것이 바람직하며
# 비즈니스 절차에 자체에 대한 고려가 필요하고 마지막으로 현실적인 상황에서의 이점과 솔루션 구현의 비용을 고려해 가치가 있는 지를 판단해야 함
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import NormalIndPower
hist_data_df = pd.read_csv("DataSet/chap8-historical_data.csv", encoding="utf-8-sig")
exp_data_df = pd.read_csv("DataSet/chap8-experimental_data.csv", encoding="utf-8-sig")
# 데이터 로드 : 전체 호텔 예약 데이터 모집단 & 50 : 50 무작위 배정된 표본
OUTCOME_COL = "booked"
TREAT_COL = "oneclick"
# 주요 컬럼명
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
# 이때 하나의 비즈니스 목표 달성을 위해 다른 비즈니스 목표를 적절히 희생해야하는 트레이드 오프가 발생할 수 있음, 이러한 경우 가중 평균 지표를
# 목표변수로 사용할 수 있으나(예를 들어 0.5 * 호텔 일일 예약 수익 + 0.5 * 고객 만족도) 가중 평균의 구조적 결함에 노출될 가능성 큼 : 가중치
# 결정 자체가 자의적이어서 가중치를 설정할때 특정 이해관계자의 판단이 반영되어 객관성 확보가 어려움, 또한 트레이프 오프 관계가 단순히 선형적으
# 로만 표현되므로 실질적으로 감수해야하는 리스크 차이가 매우 큰 상황이 동일한 크기의 지표로 표현됨, 예를 들어 '수익이 크게 증가했지만 고객 만
# 족도가 크게 감소한 경우' 와 '수익이 약간 증가했고 고객 만족도도 약간 감소한 경우' 는 전자의 리스크가 훨씬 크지만 각 상황의 가중 평균 지표는
# 동일, 따라서 실제로 얼마나 고객 만족도를 희생하고 있는지를 인지하지 못 하게 되어 잘못된 의사결정이 이뤄질 가능성 높음, 마지막으로 단기적 비
# 즈니스 목표와 장기적 비즈니스 목표의 가치 차이를 무시하게될 가능성이 존재함, 단기 매출은 올라갔지만 장기 고객 가치가 훼손되었을때 고객 가치
# 의 훼손이 적더라도 비즈니스에는 더 치명적일 수 있음 => 목표 지표를 개선하는 과정에서 허용할 수 없는 수준으로 나빠지면 안 되는 지표를 가드레
# 일 지표로 설정하고 목표 지표는 하나의 단일한 지표만 선택하는 것이 더 효과적

# %% 단순 무작위 배정의 원리
# 고객이 상품에 접근할때마다 특정 확률로 대조군에 할당하고 특정 확률로 실험군에 할당, 일반적으로 표본을 50 : 50 으로 나누어 할당하나 새로운
# 기능이 매출이나 사용자 경험을 크게 해칠 위험이 있거나 안전성이 어느 정도 확인된 상태에서 대규모 이벤트 기간처럼 실제 효과를 넓게 적용하면서
# 도 일부 비교군은 남겨야 하는 경우 다른 비율을 고려해볼 수 있음
K = 2
assgnt = np.random.uniform(0,1,1)
group = "control" if assgnt <= 1/K else "treatment"
print(group)
# 무작위 배정은 간단하게 구현 가능하나 배정의 '시점' 과 '수준' 을 세심하게 고려하여야함, '시점' 이란 관찰 단위를 실험군/대조군에 어느 순간에
# 배정하는 가를 뜻하며 배정 시점이 너무 늦으면 이미 실험 자극에 일부노출된 사람만 들어오게 되어 선택 편향이 생길 수 있고 실험 시작 전에미리
# 배정하면 이후 탈락이 집단별로 다르게 나타날 수 있어 전체 사용자에 대한 효과를 보기 어려움, '수준' 이란 관찰 단위의 범위 또는 개입이 올바른
# 행동 수준에서 발생하는 지 여부를 뜻하며 예를 들어 웹사이트를 나갔다 들어온 고객을 여전히 똑같은 그룹에 배정해야하는지가 이에 해당됨, 정답은
# 없으며 처치(개입)이 적용되는 방식과 처치가 오염될 가능성을 기준으로 정해야 함, 일반적으로 시점 관점에서는 편향을 방지하기 위해 관심 대상이
# 되는 행동을 고객이 실행하기 직전에 배정하는 것이 바람직하며 한 관찰 단위가 여러 조건에 동시에 노출되지 않도록 해야함 또한 층화 무작위 배정
# 을 사용하여 집단간 균형을 맞추는 것도 권장됨 또한 분석자의 임의가 아닌 사전에 고정된 절차로 자동 배정하는 것이 좋으며 마지막으로 배정 이후
# 탈락이 집단마다 다르게 발생하는지도 점검하는 것이 좋음
# 예를들어 교실의 급훈이 학생 학업 성취도에 영향을 주는 가를 실험하고자 할때는 급훈을 교실에 적용하기 직전에 배정이 이뤄져야 하며 개별 학생별
# 로 집단 배정 시 급훈이 적용된 처치와 그렇지 않은 처치가 섞이게 되므로 학급단위로 배정해야 하며, 학급별로 학생의 인구통계적 변수 및 환경 변
# 수도 적절하게 균형을 맞추어야하고 교사의 임의가 아닌 완전 무작위와 같은 방법으로 배정하는 것이 권장됨

# %% 실험 기본 설정 + 기저 전환율 추정 + 분석적 표본 크기 계산
# 실험 설계 시 표본을 대조군과 실험군으로 분배하며 이에 따라 두 가지 가설을 설정 할 수 있음 : 귀무가설(H₀, 원클릭 버튼이 예약 전환율에 아무
# 런 영향을 미치지 않는다) vs 대립가설(H₁, 원클릭 버튼이 실제로 전환율을 변동시킨다) -> 실험 설계 목적은 수집된 데이터를 통해 귀무가설을 기
# 각할 충분한 근거가 있는지 판단하는 것, 이때 표본의 크기에 따라 계산 비용/통계적 오류의 가능성/탐지가능한 효과의 크기 등이 달라지고 결과적으
# 로 귀무가설의 기각가능성이 달라므로 통계적 오류의 허용 범위와 탐지하고자 하는 효과의 크기 사이의 균형을 수학적으로 계산할 필요가 있음, 표본
# 크기 계산은 '목표 변수가 이항 비율일 경우' 아래 변수에 원하는 값을 입력 후 필요 표본수를 역산하여 결정할 수 있음
# α (유의수준, P-value) : 귀무가설이 실제로 참인데도 불구하고 잘못 기각할 확률 = 1종 오류, 원클릭 버튼이 사실 아무 효과가 없음에도 불구하
# 고 데이터의 우연한 편향 때문에 효과가 있다고 잘못 결론 내리는 것, α를 낮게 설정할수록 거짓 양성(false positive)을 줄일 수 있지만 반대로
# 2종 오류 위험이 높아짐, 통상 0.05로 설정
# β : 귀무가설이 실제로 거짓인데도 이를 기각하지 못할 확률 = 2종 오류, 원클릭 버튼이 실제로 전환율을 높이는 효과가 있는데도 데이터가 그것을
# 포착하지 못해 효과가 없다고 잘못 결론 내리는 것, β를 낮게 설정할수록 거짓 음성(false negative)을 줄일 수 있지만 반대로 1종 오류 위험이
# 높아짐, 통상적으로 0.2 까지 허용
# 검정력 : 1 - β, 진짜 효과가 존재할 때 그것을 올바르게 탐지할 확률, 원클릭 버튼이 실제로 전환율을 높이는 효과가 있다면 100번의 실험 중 N
# 번은 그 효과를 통계적으로 유의하게 탐지할 수 있음, 검정력이 낮은 실험은 실제로 효과가 있는 개선안을 채택하지 못하고 포기하게 됨, 반대로 검
# 정력을 지나치게 높게 설정하면 필요 표본 크기가 급격히 커져 실험 비용과 기간이 비현실적으로 증가, 통상적으로 검정력 0.8을 최소 기준 0.95를
# 엄밀한 기준으로 설정
# MDE (최소 탐지 가능 효과) : 비즈니스적으로 의미 있다고 판단하여 반드시 탐지해내고자 하는 효과의 크기 임계값, 호텔의 예약 전환율이 5% 라고
# 가정 시 MDE를 2%p로 설정하면 5%에서 7% 이상으로 오르는 변화만 탐지 대상으로 삼는 것, MDE가 작을수록 두 집단 간 분포가 크게 겹치기 때문에
# 미세한 차이를 구분하기 위해 훨씬 많은 데이터가 필요, 이때 통계적 기준이 아니라 비즈니스 판단을 통해 결정, 예를 들어 원클릭 버튼 개발 비용이
# 크다면 최소 2%p 이상의 전환율 향상이 있어야 투자 가치가 있다고 판단하는 등
# 기저 전환율 : 과거 데이터(실험 직전 데이터)에서 추정된 목표 변수 값, 예를 들어 현재 호텔 사이트에서 측정된 예약 전환율이 이에 해당, 전환율
# 이 50%에 가까울수록 분산이 가장 커져 표본 요구량이 증가함
ALPHA = 0.05 # 유의수준
POWER = 0.80 # 검정력
MDE = 0.01 # 최소 탐지 가능 효과(절댓값, %p)
ALTERNATIVE = "larger" # 검정 방향(양방향 or 단방향)
baseline_cr = hist_data_df[OUTCOME_COL].mean() # 기저 전환율
target_cr = baseline_cr + MDE # 목표 전환율
effect_size = proportion_effectsize(target_cr, baseline_cr)
# 두 비율의 차이를 표준화한 효과 크기(Cohen's h) 계산, 비율 데이터는 분산이 전환율 크기에 따라 달라지므로 표준화된 크기 사용해야 더 안정적
power_analysis = NormalIndPower()
required_n_per_group = int(np.ceil(
    power_analysis.solve_power(
        effect_size=effect_size,
        alpha=ALPHA,
        power=POWER,
        alternative=ALTERNATIVE
    )
))
required_n_total = required_n_per_group * 2
print("\n[Part 2] 분석적 표본 크기 계산")
print(f"- 과거 데이터 수            : {len(hist_data_df):,}")
print(f"- 기저 전환율              : {baseline_cr:.4f} ({baseline_cr*100:.2f}%)")
print(f"- 목표 전환율              : {target_cr:.4f} ({target_cr*100:.2f}%)")
print(f"- Cohen's h                : {effect_size:.4f}")
print(f"- 필요 표본 수(그룹당)     : {required_n_per_group:,}")
print(f"- 필요 표본 수(전체)       : {required_n_total:,}") # 필요 표본 수 계산
# 단, 목표 변수가 이항 비율일 경우에만 Closed-Form Solution으로 계산할 수 있으며 표본 크기를 계산하는 Closed-Form Solution이 존재하지
# 않는 경우 부트스트랩 기반 시뮬레이션으로 검정력 검증이 필요함

# %% 사전 실험 효과성 검증
# 위 과정에서 무작위 배정된 표본 데이터에 대해 실험이 진행되고 난 뒤에 사후에 그 효과성을 측정할 수 있음 그러나 실험이 진행된 뒤에는 이미 시
# 간과 비용이 투입된 뒤이기 때문에 사후 검증을 통해서는 소비된 시간과 비용을 되돌릴 수 없음 따라서 실험을 진행하기 전 데이터 파이프라인의 오
# 류/집단 간 사전 편향/측정 방식의 문제 등 실험 설계 자체의 신뢰성 검증 필요
PLACEBO_NSIM = 500
PLACEBO_TOL = 0.02
placebo_df = hist_data_df.copy()
placebo_df["fake_group"] = rng.binomial(1, 0.5, size=len(placebo_df))
placebo_control = placebo_df[placebo_df["fake_group"] == 0]
placebo_treat = placebo_df[placebo_df["fake_group"] == 1]
pc_booked = int(placebo_control[OUTCOME_COL].sum())
pc_not = len(placebo_control) - pc_booked
pt_booked = int(placebo_treat[OUTCOME_COL].sum())
pt_not = len(placebo_treat) - pt_booked
placebo_ct = [[pc_booked, pc_not], [pt_booked, pt_not]]
chi2_stat, p_chi2, _, expected = chi2_contingency(placebo_ct)
if expected.min() < 5:
    _, placebo_p = fisher_exact(placebo_ct)
    placebo_test = "Fisher exact"
else:
    placebo_p = p_chi2
    placebo_test = "Chi-square"
print("\n[Part 3] 단일 플라시보 테스트")
print(f"- 가짜 대조군 전환율       : {pc_booked / len(placebo_control):.4f}")
print(f"- 가짜 실험군 전환율       : {pt_booked / len(placebo_treat):.4f}")
print(f"- 사용 검정                : {placebo_test}")
print(f"- p-value                  : {placebo_p:.4f}")
# 플라시보 테스트 : 실험이 시작되기전 과거 데이터에 A/B 테스트 분석 코드를 그대로 적용, 이는 실제 유의미한 개입이 아니므로 대조군/실험군 간
# 효과가 유의미하게 검출되면 실험 또는 데이터 편향 의심 => 가짜 대조군/가짜 실험군 간의 전환율을 카이 제곱 검정
placebo_pvals = []
for i in range(PLACEBO_NSIM):
    fake = hist_data_df.copy()
    fake["fake_group"] = np.random.default_rng(i).binomial(1, 0.5, size=len(fake))
    c = fake[fake["fake_group"] == 0]
    t = fake[fake["fake_group"] == 1]
    cb = int(c[OUTCOME_COL].sum())
    cn = len(c) - cb
    tb = int(t[OUTCOME_COL].sum())
    tn = len(t) - tb
    ct = [[cb, cn], [tb, tn]]
    _, p, _, ex = chi2_contingency(ct)
    if ex.min() < 5:
        _, p = fisher_exact(ct)
    placebo_pvals.append(p)
placebo_pvals = np.array(placebo_pvals)
false_pos_rate = (placebo_pvals < ALPHA).mean()
placebo_ok = abs(false_pos_rate - ALPHA) <= PLACEBO_TOL
print("\n[Part 3] 반복 플라시보 테스트")
print(f"- 반복 횟수                : {PLACEBO_NSIM}")
print(f"- 관측 1종 오류율          : {false_pos_rate:.4f}")
print(f"- 기대 1종 오류율(alpha)   : {ALPHA:.4f}")
print(f"- 신뢰성 판정              : {'정상' if placebo_ok else '재점검 필요'}")
# 반복 시뮬레이션 : N번 반복 후 카이제곱 검정의 p-value 분포가 균등한지 확인, 카이제곱 검정의 결과가 단 한번의 단순한 우연이 아님을 확인
# → 1종 오류율이 α 근방이면 정상 현저히 높으면 구조적 문제 의심

# %% 표본 추출 + 무작위 배정 + 표본 검정력 재확인
# 실험 설계 과정에 문제가 없으면 전체 모집단으로부터 필요 표본수 만큼 추출 후 단순 무작위 배정 실시 후 추출된 표본의 검정력 재확인, 표본 추
# 출 과정에서 중간확인을 지나치게 자주하고 P-value가 유의해 보이는 순간 멈추는 행동 등은 지양, 분석자가 우연한 변동을 진짜 효과처럼 보게될
# 가능성 만들어 1종 오류 확률 크게 높임
replace = required_n_total > len(hist_data_df)
sim_exp_df = hist_data_df.sample(
    n=required_n_total,
    replace=replace,
    random_state=RANDOM_SEED
).copy()
# 필요 표본수보다 모집단 크기가 작으면 복원추출하고 아니면 비복원추출
sim_exp_df[TREAT_COL] = rng.binomial(1, 0.5, size=len(sim_exp_df))
control_df = sim_exp_df[sim_exp_df[TREAT_COL] == 0]
treatment_df = sim_exp_df[sim_exp_df[TREAT_COL] == 1]
# 50:50 단순 무작위 배정
n_control = len(control_df)
n_treatment = len(treatment_df)
n_min = min(n_control, n_treatment)
achieved_power = power_analysis.solve_power(
    effect_size=effect_size,
    alpha=ALPHA,
    nobs1=n_min,
    alternative=ALTERNATIVE
)
print("\n[Part 4] 직접 추출 표본의 무작위 배정 결과")
print(f"- 추출 방식                : {'복원추출' if replace else '비복원추출'}")
print(f"- 전체 표본 수             : {len(sim_exp_df):,}")
print(f"- 대조군 표본 수           : {n_control:,}")
print(f"- 실험군 표본 수           : {n_treatment:,}")
print(f"- 현재 기준 그룹 표본 수   : {n_min:,}")
print(f"- 현재 달성 검정력         : {achieved_power:.4f} ({achieved_power*100:.2f}%)")

# %% 실험 결과 분석 - 단순 수치 비교
control_df = exp_data_df[exp_data_df[TREAT_COL] == 0].copy()
treatment_df = exp_data_df[exp_data_df[TREAT_COL] == 1].copy()
n_control = len(control_df)
n_treatment = len(treatment_df)
n_min = min(n_control, n_treatment) # exp_data_df 가 실험이 종료된 표본이라고 가정
ctrl_booked = int(control_df[OUTCOME_COL].sum())
trt_booked = int(treatment_df[OUTCOME_COL].sum())
cr_control = ctrl_booked / n_control
cr_treatment = trt_booked / n_treatment
raw_diff = cr_treatment - cr_control
contingency = [
    [ctrl_booked, n_control - ctrl_booked],
    [trt_booked, n_treatment - trt_booked]
]
chi2_stat, p_chi2, _, expected = chi2_contingency(contingency)
print("\n[Part 5] 실험 결과 분석 - 카이제곱 검정")
print(f"- 대조군 전환율            : {cr_control:.4f} ({cr_control*100:.2f}%)")
print(f"- 실험군 전환율            : {cr_treatment:.4f} ({cr_treatment*100:.2f}%)")
print(f"- 관측 전환율 차이         : {raw_diff:.4f} ({raw_diff*100:.2f}%p)")
print(f"- p-value                  : {p_chi2:.4f}")
# 무작위 배정된 표본 데이터에 대해 카이 제곱 검정 : 귀무가설이 참이라면 각 칸에 기대되는 빈도와 실제 관측된 빈도를 비교해 두 행의 비율이 같
# 다 즉 개입이 유의미한 목표 변수에 유의미한 영향을 미치지 못 했다는 귀무가설을 검정, p값이 유의수준 미만이면 귀무가설을 기각
se_diff = np.sqrt(
    cr_control * (1 - cr_control) / n_control +
    cr_treatment * (1 - cr_treatment) / n_treatment
)
z_critical = 1.96
ci_lower = raw_diff - z_critical * se_diff
ci_upper = raw_diff + z_critical * se_diff
print("\n[Part 5] 실험 결과 분석 - 전환율 차이의 신뢰구간")
print(f"- 95% CI                   : [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"- 사전 설정 MDE            : {MDE:.4f}")
# 대조군과 실험군의 전환율 차이의 신뢰구간 도출 : 0을 포함하는가 즉 효과가 없을 가능성을 배제할 수 있는가, 사전에 정한 MDE를 포함하거나 넘는
# 가 즉 사업적으로 의미있는 수준인가, 구간이 지나치게 넓은가 즉 추정이 아직 불안정한가를 확인 가능, P-value 보다 풍부한 의사결정 근거 제공

# %% 실험 결과 분석 - 공변량 보정과 정밀한 해석
# 공변량 보정 로지스틱 회귀 : 관심 변수의 효과를 다른 변수들의 영향을 함께 고려하면서 추정, 무작위 배정 실험이라 하더라도 나이/성별/기간 같
# 은 변수는 예약 여부와 밀접하게 관련될 수 있음, 로지스틱 회귀는 다른 변수들로 인한 추정의 분산을 줄여 더 정밀한 결과를 제공
candidate_covars = ["age", "gender", "period"]
adj_covars = [c for c in candidate_covars if c in exp_data_df.columns]
def _term(c, df):
    return f"C({c})" if (df[c].dtype == "object" or str(df[c].dtype).startswith("category")) else c
adj_formula = f"{OUTCOME_COL} ~ {TREAT_COL}"
if adj_covars:
    adj_formula += " + " + " + ".join(_term(c, exp_data_df) for c in adj_covars)
logit_res = smf.logit(adj_formula, data=exp_data_df).fit(disp=0)
coef = logit_res.params[TREAT_COL]
p_adj = logit_res.pvalues[TREAT_COL]
ci_adj_low, ci_adj_high = logit_res.conf_int().loc[TREAT_COL]
odds_ratio = np.exp(coef)
print("\n[Part 6] 공변량 보정 로지스틱 회귀")
print(f"- 회귀식                   : {adj_formula}")
print(f"- oneclick 계수            : {coef:.4f}")
print(f"- oneclick p-value         : {p_adj:.4f}")
print(f"- oneclick 95% CI          : [{ci_adj_low:.4f}, {ci_adj_high:.4f}]")
print(f"- odds ratio               : {odds_ratio:.4f}")
# 다만 로지스틱 회귀 계수는 변수간 영향을 1 대 1로 나타내지 않으므로 직관적이지 않음, 따라서 모형이 예측하는 평균 예약확률 차이를 함께 산출
# 하면 더 직관적, 같은 표본에 대해 oneclick = 0인 경우와 oneclick = 1인 경우의 예측확률을 각각 계산해 평균 예약확률 차이(평균 몇 %p 상
# 승하는가)를 구하면 더 직관적
no_button_df = exp_data_df.copy()
no_button_df[TREAT_COL] = 0
button_df = exp_data_df.copy()
button_df[TREAT_COL] = 1
pred0 = logit_res.predict(no_button_df)
pred1 = logit_res.predict(button_df)
avg_prob_diff = (pred1 - pred0).mean()
print("\n[Part 7] 평균 예측 확률 차이")
print(f"- 평균 예약확률 차이       : {avg_prob_diff:.4f} ({avg_prob_diff*100:.2f}%p)")
# 평균 예측 확률 차이에 대한 부트스트랩 신뢰구간을 도출하여 0을 포함하는가, 사전에 정한 MDE를 포함하거나 넘는가, 구간이 지나치게 넓은가 확인
BOOT_B = 300
BOOT_CONF = 0.95
RANDOM_SEED = 42
def prob_diff_metric(dat_df):
    fit_formula = f"{OUTCOME_COL} ~ {TREAT_COL}"
    if adj_covars:
        fit_formula += " + " + " + ".join(_term(c, dat_df) for c in adj_covars)
    res = smf.logit(fit_formula, data=dat_df).fit(disp=0)
    d0 = dat_df.copy()
    d1 = dat_df.copy()
    d0[TREAT_COL] = 0
    d1[TREAT_COL] = 1
    return (res.predict(d1) - res.predict(d0)).mean()
boot_vals = []
boot_rng = np.random.default_rng(RANDOM_SEED)
for _ in range(BOOT_B):
    boot_df = exp_data_df.sample(
        n=len(exp_data_df),
        replace=True,
        random_state=int(boot_rng.integers(0, 10**9))
    )
    boot_vals.append(prob_diff_metric(boot_df))
boot_vals = np.sort(np.array(boot_vals))
alpha_tail = (1 - BOOT_CONF) / 2
ci_boot_low = np.quantile(boot_vals, alpha_tail)
ci_boot_high = np.quantile(boot_vals, 1 - alpha_tail)
print("\n[Part 7] 평균 예측 확률 차이 - Bootstrap 신뢰구간")
print(f"- bootstrap 반복 수        : {BOOT_B}")
print(f"- 평균 확률 차이 추정치    : {avg_prob_diff:.4f}")
print(f"- {int(BOOT_CONF*100)}% bootstrap CI : [{ci_boot_low:.4f}, {ci_boot_high:.4f}]")

# %% 실험 종료 후 최종 의사 결정
# 단순 비율 비교에서 유의한가 & 공변량 보정 후에도 유의한가 & 평균 효과가 MDE 이상인가 & 부트스트랩 신뢰구간이 0을 배제하는가를 바탕으로 해
# 당 개입의 효과성 및 실험의 지속 여부를 결정
is_sig_primary = p_chi2 < ALPHA
is_sig_adjusted = p_adj < ALPHA
is_business_meaningful = avg_prob_diff >= MDE
ci_excludes_zero = ci_boot_low > 0
ci_includes_mde = ci_boot_low <= MDE <= ci_boot_high
if is_sig_primary and is_sig_adjusted and ci_excludes_zero and is_business_meaningful:
    final_verdict = "채택 권고: 통계적으로 유의하고 효과 크기도 실무 기준 충족"
elif is_sig_primary and is_sig_adjusted and ci_excludes_zero and not is_business_meaningful:
    final_verdict = "보류: 효과는 유의하지만 MDE 미만"
elif (not is_sig_primary or not is_sig_adjusted) and ci_includes_mde:
    final_verdict = "보류: 표본 부족 가능성, 추가 데이터 검토"
else:
    final_verdict = "기각: 유의성 또는 실질 효과 근거가 부족"
print("\n[Part 8] 최종 의사결정")
print(f"- 1차 분석 유의성         : {is_sig_primary}")
print(f"- 보정 분석 유의성         : {is_sig_adjusted}")
print(f"- 효과의 실무적 의미       : {is_business_meaningful}")
print(f"- CI가 0 초과 여부         : {ci_excludes_zero}")
print(f"- CI의 MDE 포함 여부       : {ci_includes_mde}")
print(f"- 최종 판단                : {final_verdict}")