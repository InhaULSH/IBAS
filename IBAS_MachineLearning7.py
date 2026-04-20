# 11 - 1. 텍스트 분석 개요
# 비정형 데이터인 텍스트에서 의미 있는 정보를 추출하고 가공, 비정형 텍스트 데이터를 어떻게 숫자 형태의 피처로 변환(피처 벡터화)하느냐가 핵심

# %% 텍스트 데이터 전처리 - 원본 텍스트는 정제되지 않은 상태이므로 클렌징, 토큰화, 스톱워드 제거 등 텍스트를 분석하기 좋게 다듬는 과정이 필요
import nltk
nltk.download('punkt_tab')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
# 전처리에 필요한 NLTK 데이터셋 다운로드 (최초 1회만 필요)
nltk.download('punkt') # 토크나이저
nltk.download('stopwords') # 불용어 사전
nltk.download('wordnet') # 표제어 추출(Lemmatization) 사전
nltk.download('omw-1.4')
# 1. 예제 텍스트 데이터
text_data = "The quick brown fox jumped over the lazy dogs. Text analysis is fascinating!"
# 2. 클렌징 : HTML 태그, 특수문자, 이모티콘 등 분석에 불필요한 노이즈를 제거, 대/소문자를 통일하는 작업도 포함
text_lower = text_data.lower()
print(f"원본: {text_data}")
print(f"소문자 변환: {text_lower}")
# 3. 토큰화 : 긴 텍스트를 분석 가능한 작은 단위(문장, 단어 등)로 자름
tokens = word_tokenize(text_lower)
print(f"토큰화 결과: {tokens}")
# 4. 스톱워드 제거 : 'the', 'is', 'a' 처럼 문법적으로는 필요하지만 문맥상 큰 의미가 없는 단어(불용어)를 제거
# 영어 불용어 목록 가져오기
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words and word.isalnum()]
# .isalnum()은 특수문자(마침표 등) 제거를 위해 추가
print(f"불용어 제거 후: {filtered_tokens}")
# 5. 어근 추출 : 단어의 형태가 변형된 경우(예: working, worked), 이를 원형(work)으로 통일
# Stemming: 단순히 단어의 어미를 자르는 방식 -> 높은 속도, 상대적으로 낮은 품질
# Lemmatization: 문법적인 요소와 의미를 고려하여 정확한 원형을 찾는 방식 -> 상대적으로 우수한 품질, 느린 속도
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()
# 예시 단어들로 비교
words = ['working', 'works', 'jumped', 'happier', 'am']
print("\n[어근 추출 비교]")
for word in words:
    print(f"단어: {word:10} | Stemming: {stemmer.stem(word):10} | Lemmatization: {lemmatizer.lemmatize(word, pos='v')}")
    # pos='v'는 동사(verb)로 간주하고 원형을 찾으라는 옵션

# %% 피처 벡터화 - BOW : 토큰의 문맥이나 순서는 무시하고, 오로지 어떤 단어가 몇 번 나왔는가에만 집중, 쉽고 빠르지만 문맥 의미는 손실됨
# 카운트 기반 벡터화 : 단순히 해당 단어가 문서에 몇 번 등장했는지 계산, 문법적으로 자주 쓰이는 무의미한 단어들까지 높은 가중치를 받게 됨
# TF-IDF : 카운트 기반의 문제점을 보완, TF(특정 문서에서 특정 단어가 등장하는 빈도) + IDF(전체 문서 집합에서 그 단어가 등장하는 빈도)
# 를 모두 고려, 특정 문서에는 자주 나오지만 다른 문서들에는 잘 안 나오는 단어가 높은 가중치를 받게 됨
# 피쳐 벡터화를 통해 원본 텍스트 데이터를 수치 기반 행렬 데이터로 변환하여야만 머신러닝 모델 학습 가능
# N-gram (단어를 하나씩 자르는 게 아니라, 두 개나 세 개씩 묶어서 토큰으로 만듦) / 워드 임베딩 (단어를 의미 공간에 배치, 비슷한 의미를
# 가진 단어는 벡터 공간에서 가깝게 위치, 딥러닝 프레임워크나 전용 라이브러리 필요) 등 문맥 의미를 활용할 수 있는 기법도 사용 가능
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
# 예제 문서 (Corpus)
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
print("--- 1. Count Vectorization (단순 카운트) ---")
# 객체 생성 및 변환
cnt_vect = CountVectorizer()
X_cnt = cnt_vect.fit_transform(corpus)
# 결과 확인 (행렬 형태를 눈으로 보기 위해 배열로 변환)
print("단어 사전(Vocabulary):", cnt_vect.get_feature_names_out())
print("변환된 행렬:\n", X_cnt.toarray())
print("\n--- 2. TF-IDF Vectorization (가중치 적용) ---")
# 객체 생성 및 변환
tfidf_vect = TfidfVectorizer()
X_tfidf = tfidf_vect.fit_transform(corpus)
# 결과 확인 (소수점 2자리까지만 출력)
print("단어 사전(Vocabulary):", tfidf_vect.get_feature_names_out())
print("변환된 행렬:\n", np.round(X_tfidf.toarray(), 2))
print("\n--- 3. N-gram 적용 ---")
# ngram_range=(1, 2) 설정: 유니그램과 바이그램을 모두 포함
ngram_vect = CountVectorizer(ngram_range=(1, 2))
X_ngram = ngram_vect.fit_transform(corpus)
# 결과 확인
print(f"단어 사전(Vocabulary) 크기: {len(ngram_vect.get_feature_names_out())}")
print("확장된 단어 사전:\n", ngram_vect.get_feature_names_out())
print("변환된 행렬:\n", np.round(X_ngram.toarray(), 2))
# %% 사이킷런의 피쳐 벡터화 클래스 : CountVectorizer와 TfidfVectorizer 클래스는 아래와 같은 파라미터로 피쳐 선택 기준을 제어 가능
# max_df : 너무 자주 등장하는 단어를 제외, 전체 문서의 특정 퍼센티지 이상에서 나타나는 단어는 스톱워드로 간주하여 제거
# min_df : 너무 적게 등장하는 단어를 제외, 전체 문서 중 특정 회수 미만으로 나타나는 단어는 중요성이 낮다고 간주하여 제거
# max_features : 추출할 피처의 최대 개수를 제한, 빈도수가 가장 높은 순으로 정렬하여, 상위 N개의 단어만 피처로 사용
# stop_words : 불용어 사전을 지정, 'english'와 같이 언어명을 지정하거나 리스트를 직접 입력
# tokenizer : 별도의 토큰화 함수(예: NLTK나 KoNLPy 함수)를 직접 지정
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
    'This is a completely unique sentence.',  # 'unique' 같은 희귀 단어 포함
]
print("--- 1. 기본 설정 (모든 단어 포함) ---")
vect_default = CountVectorizer()
vect_default.fit(corpus)
print(f"피처 개수: {len(vect_default.get_feature_names_out())}")
print(vect_default.get_feature_names_out())
print("\n--- 2. 파라미터 제어 적용 ---")
# min_df=2: 최소 2개 문서 이상 나온 단어만 선택
# max_features=3: 빈도수 상위 3개 단어만 선택
vect_param = CountVectorizer(min_df=2, max_features=3)
vect_param.fit(corpus)
print(f"피처 개수: {len(vect_param.get_feature_names_out())}")
print("선택된 최상위 단어들:", vect_param.get_feature_names_out())
# %% 피처 벡터화된 텍스트는 대부분 대부분의 값이 0으로 채워지는 희소행렬이 생성됨, 희소행렬을 그대로 처리하면 메모리 낭비 및 계산 비효율
# 문제가 발생하므로 특수한 저장 방법 필요 => COO (0이 아닌 데이터만 (행, 열, 값)의 형태로 저장) / CSR (COO 방식에서 행 인덱스 정보를
# 압축하여 저장, COO보다 메모리 효율성 및 연산속도 높음) : 사이킷런은 기본적으로 CSR을 사용하여 희소행렬을 저장
import numpy as np
from scipy import sparse
dense_matrix = np.array([[3, 0, 1],
                         [0, 2, 0],
                         [0, 0, 0],
                         [0, 0, 0],
                         [1, 0, 0],
                         [0, 0, 0]])
# 0이 아닌 데이터 추출
data = np.array([3, 1, 2, 1]) # 실제 값
row_pos = np.array([0, 0, 1, 4]) # 행 위치
col_pos = np.array([0, 2, 1, 0]) # 열 위치
# COO 형식으로 변환
coo = sparse.coo_matrix((data, (row_pos, col_pos)))
print("--- COO 변환 결과 (메모리 주소 대신 내용 출력) ---")
print(coo)
# 4. CSR 형식으로 변환 (COO보다 더 효율적)
csr = coo.tocsr() # COO 객체를 CSR로 변환
print("\n--- CSR 변환 결과 ---")
print(csr)

# %% 머신러닝 모델링 - 피쳐 벡터화된 희소행렬을 사용하여 머신러닝 모델 학습, 예측, 평가 수행
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# 1. 학습(Train) 데이터와 테스트(Test) 데이터 로드
# remove=('headers', 'footers', 'quotes') 옵션으로 본문만 추출
train_news = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), random_state=156)
X_train = train_news.data
y_train = train_news.target
test_news = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), random_state=156)
X_test = test_news.data
y_test = test_news.target
print(f"학습 데이터 크기: {len(train_news.data)}")
print(f"테스트 데이터 크기: {len(test_news.data)}")
# 2. CountVectorizer를 이용한 피처 벡터화 변환
cnt_vect = CountVectorizer()
# 학습 데이터로 피처(단어) 사전을 만들고(fit), 변환(transform)까지 수행
cnt_vect.fit(X_train)
X_train_cnt_vect = cnt_vect.transform(X_train)
# 학습 데이터에서 만들어진 피처 사전을 기준으로 테스트 데이터 변환
# 주의: 테스트 데이터에는 절대 .fit()을 호출하면 안 됨, 학습 데이터로 만든 단어 사전의 기준이 무시되기 때문
X_test_cnt_vect = cnt_vect.transform(X_test)
print(f"학습 데이터 벡터 Shape: {X_train_cnt_vect.shape}")
# 3. 로지스틱 회귀 모델 학습 및 예측, 평가
lr_clf = LogisticRegression(solver="lbfgs")
lr_clf.fit(X_train_cnt_vect, y_train)
preds = lr_clf.predict(X_test_cnt_vect)
print(f'CountVectorizer를 이용한 로지스틱 회귀의 정확도: {accuracy_score(y_test, preds):.3f}')

# 11 - 2. 다양한 텍스트 분석 기법
# %% 감성 분석 - 텍스트에 나타난 사람의 주관적인 의견/감정/태도 등을 분석하는 것, 기본적으로는 지도 학습, 지도학습은 이 문장은 긍정이라는
# 정답이 필요하지만, 현실에서는 정답이 없는 데이터가 더 많음 => 감성어휘 사전 기반 감성 분석 : 각 단어별로 긍정/부정 점수를 매겨두고 문장
# 속 단어들의 점수를 합산하여 최종 감성을 판별, 비지도 학습 기반 감성 분석 가능, 감성어휘 사전 자체 구축하는 방법도 가능
# 주요 감성어휘 사전 : SentiWordNet (WordNet 기반) / VADER (소셜 미디어 같은 짧은 텍스트나 리뷰 분석에 최적화) 등
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score
import nltk
# 1. VADER 사전 및 필수 자원 다운로드
nltk.download('vader_lexicon')
# 2. 데이터 로드
review_df = pd.read_csv(
    "DataSet/IMDBdataset.csv",
    sep=",",
    quotechar='"',
    encoding="utf-8",
)
# 결측치 제거
review_df = review_df.dropna(subset=['review'])
# 3. 레이블 수치화 (문자열 'positive'/'negative'를 1/0으로 변환)
# VADER의 결과값과 비교하기 위해 반드시 필요합니다.
review_df['sentiment'] = review_df['sentiment'].map({'positive': 1, 'negative': 0})
# 4. VADER 객체 생성
vader_analyzer = SentimentIntensityAnalyzer()
# 5. 감성 점수 계산 함수 정의
def get_vader_sentiment(review, threshold=0.1):
    # 데이터가 문자열이 아닌 경우를 대비해 str() 처리
    scores = vader_analyzer.polarity_scores(str(review))
    compound_score = scores['compound']
    # 임계값(threshold)보다 크면 긍정(1), 작으면 부정(0)
    return 1 if compound_score >= threshold else 0
# 6. 전체 리뷰에 적용
review_df['vader_preds'] = review_df['review'].apply(lambda x: get_vader_sentiment(x))
# 7. 성능 평가
y_target = review_df['sentiment']
vader_preds = review_df['vader_preds']
print("-" * 30)
print(f'VADER 감성 분석 정확도: {accuracy_score(y_target, vader_preds):.4f}')
print("-" * 30)
# 분석 결과 샘플 확인
print(review_df[['review', 'sentiment', 'vader_preds']].head())

# %% 토픽 모델링 - 문서 집합에 숨어 있는 주제(Topic)를 찾아내는 비지도 학습 => LDA : 한 문서는 여러 개의 주제가 섞여서 만들어지며 한 주제
# 는 같이 등장하는 여러 단어들의 분포로 이뤄져 있다고 가정, 단어들의 분포를 계산하여 문서가 어떤 주제들에 속해 있는지 확률적으로 추론
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
# 1. 데이터 로드
# remove 파라미터로 순수 본문만 남긴 뒤 8개 카테고리만 우선 추출
cats = ['rec.motorcycles', 'rec.sport.baseball', 'comp.graphics', 'comp.windows.x',
        'talk.politics.mideast', 'soc.religion.christian', 'sci.electronics', 'sci.med']
news_df = fetch_20newsgroups(subset='all', categories=cats,
                             remove=('headers', 'footers', 'quotes'),
                             random_state=0)
# 2. CountVectorizer 객체 생성 및 변환
# max_df=0.95: 전체 문서의 95% 이상에 나타나는 너무 흔한 단어 제외
# max_features=1000: 빈도수 상위 1,000개 단어만 추출
count_vect = CountVectorizer(max_df=0.95, max_features=1000, min_df=2, stop_words='english', ngram_range=(1,2))
feat_vect = count_vect.fit_transform(news_df.data)
print(f'CountVectorizer Shape: {feat_vect.shape}')
lda = LatentDirichletAllocation(n_components=8, random_state=0)
lda.fit(feat_vect)
print(lda.components_.shape)
# 3. 토픽별 주요 단어 확인
def display_topics(model, feature_names, no_top_words):
    # 각 토픽별(topic_idx)로 반복
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx}")
        # topic 배열에서 값이 큰 순서대로 정렬한 후, 뒤에서부터 no_top_words 개수만큼 인덱스 가져오기
        # 예: [1, 10, 5] -> argsort -> [0, 2, 1] -> 뒤에서 2개 -> [1, 2] (값 10, 5의 인덱스)
        topic_word_indexes = topic.argsort()[:-no_top_words - 1:-1]
        # 인덱스를 실제 단어 문자열로 변환하여 결합
        top_words_str = " ".join([feature_names[i] for i in topic_word_indexes])
        print(top_words_str)
# CountVectorizer에서 단어 이름(feature names) 가져오기
feature_names = count_vect.get_feature_names_out()
# 토픽별 상위 10개 단어 출력
display_topics(lda, feature_names, 10)

# %% 문서 군집화 - 정답 없이 내용이 서로 비슷한 문서들끼리 그룹화하는 판단하는 비지도학습 기법, 지도 학습 기법인 텍스트 분류와는 다름
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
# 1. 명확한 구분을 위해 3개의 카테고리만 지정
categories = ['sci.space', 'rec.autos', 'sci.med']
# 2. 데이터 로드
dataset = fetch_20newsgroups(subset='all', categories=categories,
                             remove=('headers', 'footers', 'quotes'),
                             random_state=0)
# 3. 데이터 확인 (DataFrame 변환)
df = pd.DataFrame({'content': dataset.data})
print(f"전체 문서 개수: {len(df)}")
print(df.head())
# 4. TF-IDF 벡터화
# max_df=0.5: 50% 이상의 문서에서 나타나는 흔한 단어 제외
# min_df=2: 최소 2번 이상 등장하는 단어만 포함
# stop_words='english': 영어 불용어 제거
tfidf_vect = TfidfVectorizer(tokenizer=None, stop_words='english',
                             ngram_range=(1,2), min_df=2, max_df=0.5)
feature_vect = tfidf_vect.fit_transform(df['content'])
print(f"TF-IDF 행렬 크기: {feature_vect.shape}")
# 5. K-Means 군집화
# 3개의 주제(우주, 자동차, 의학)가 있다고 가정하고 3개 그룹으로 나눔
km_cluster = KMeans(n_clusters=3, max_iter=10000, random_state=0)
km_cluster.fit(feature_vect)
df['cluster_label'] = km_cluster.labels_
print("\n[클러스터별 문서 개수]")
print(df['cluster_label'].value_counts().sort_index())
for i in range(3):
    print(f"\n--- Cluster {i} 대표 문서 내용 (일부) ---")
    # 해당 클러스터의 첫 번째 문서 200자만 출력
    print(df[df['cluster_label'] == i]['content'].iloc[0][:200])

# %% 문서 유사도 분석 - 두 문서가 구체적으로 얼마나 비슷한지를 수치로 계산, 주로 '코사인 유사도'(벡터 공간 상의 화살표로 나타내었을 때 두
# 화살표 사이의 각도를 측정)을 사용, 문서의 길이에 영향을 받지 않고 단어 패턴의 유사도를 측정할 수 있으며 희소 행렬 처리에도 유리하기 때문
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
# 1. 간단한 예제 문서
doc_list = [
    'if you take the blue pill, the story ends',
    'if you take the red pill, you stay in wonderland',
    'if you take the red pill, I show you how deep the rabbit hole goes'
]
# 2. TF-IDF 벡터화
tfidf_vect = TfidfVectorizer()
feature_vect = tfidf_vect.fit_transform(doc_list)
# 3. 코사인 유사도 계산
# 첫 번째 문서와 두 번째 문서의 유사도 비교
similarity_simple = cosine_similarity(feature_vect[0], feature_vect[1])
print(f'문서 1과 문서 2의 코사인 유사도: {similarity_simple}')
# 4. 전체 문서 간의 유사도 행렬 계산
# 모든 문서 쌍(Pair)에 대한 유사도를 한 번에 구함
similarity_pair = cosine_similarity(feature_vect, feature_vect)
print('\n[전체 문서 유사도 행렬]')
print(similarity_pair)
# 코사인 유사도는 문서의 전반적인 패턴만 파악하므로 데이터의 절대적인 빈도수나 수치가 큰 의미를 가질 때는 유클리드 거리(두 점 사이의 직선
# 거리)를, 빈도수는 무시하고 단어의 존재 여부가 중요한 추천 시스템의 경우 자카드 유사도(두 집합의 교집합 크기를 합집합 크기로 나눈 값)를
# 사용하는 것이 더 유리함

# %% KoNLPy 기반 한국어 텍스트 분석 - 한국어는 단어와 어미/조사를 분리하기 어려우므로 한글을 분석할 때는 이것들을 분리하여 형태소를 추출
# 하는 과정이 필수적, KoNLPy는 텍스브 분석 과정에서 이러한 형태소 분석을 쉽게 해주는 파이썬 라이브러리임
# 한국어는 상용 불용어 사전이 존재하지 않으므로 개발자가 수동으로 불용어 사전을 정의하여 불용어를 처리해야함
import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# 1. 데이터 로드
train_df = pd.read_csv('DataSet/ratings_train.txt', sep='\t')
train_df = train_df.fillna(' ')
# 2. 불용어 사전 정의
stop_words = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
# 3. 형태소 분석 및 불용어 제거 함수 정의
okt = Okt()
def tw_tokenizer(text):
    # 입력된 텍스트를 형태소 단위로 토큰화
    tokens = okt.morphs(text)
    # 토큰 중에서 불용어 리스트에 포함되지 않은 단어만 살리기
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens
# 4. TF-IDF 벡터화 객체 생성
# tokenizer에 tw_tokenizer 함수를 연결
tfidf_vect = TfidfVectorizer(tokenizer=tw_tokenizer,
                             ngram_range=(1, 2),
                             min_df=3,
                             max_df=0.9)
# 5. 벡터화 수행
print("불용어를 제거하며 벡터화 수행 중... (시간이 조금 걸릴 수 있습니다)")
tfidf_matrix_train = tfidf_vect.fit_transform(train_df['document'])
print(f"생성된 TF-IDF 행렬의 크기: {tfidf_matrix_train.shape}")
# 6. 로지스틱 회귀
lr_clf = LogisticRegression(C=3.5, random_state=0)
lr_clf.fit(tfidf_matrix_train, train_df['label'])
print("모델 학습 완료!")
# 7. 테스트 데이터 로드 및 정확도 평가
test_df = pd.read_csv('DataSet/ratings_test.txt', sep='\t')
test_df = test_df.fillna(' ')
tfidf_matrix_test = tfidf_vect.transform(test_df['document'])
preds = lr_clf.predict(tfidf_matrix_test)
print(f'TF-IDF 로지스틱 회귀 정확도: {accuracy_score(test_df["label"], preds):.4f}')