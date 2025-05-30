# 7. 비지도학습 - 추천시스템
# 롱테일 비즈니스 = 대형 소매사업, 경험상품 = OTT/스트리밍 서비스에 매우 유용

# 협업 필터링(메모리 기반 필터링)
# 사용자의 선호 패턴이 유사한 아이템 그룹 식별(아이템 기반) = 이 아이템을 선호한 고객들은 저 아이템도 선호했음
# 유사 선호를 가진 사용자 그룹 식별(사용자 기반) = 사용자와 비슷한 고객들은 저 아이템도 선호했음
# 사용자 - 아이템 상호작용 매트릭스로 변환하여 분석해야함, 대체로 아이템 기반 방식을 더 많이 사용
# 아이템의 특성과는 관계없이 사용자 기록을 기준으로 필터링 = 간결함, 모델 설명 가능, 신규 아이템 추가에 의한 영향 적음(안정성)
# 신규 데이터에 대해서 추천 불가(콜드스타트 문제), 벡터 크기가 증가할 수록 계산량 증가 (확장성 문제), Sparse 데이터 문제
import pandas as pd
import numpy as np
movies = pd.read_csv('./DataSet_MachineLearning/Visual/movies.csv')
ratings = pd.read_csv('./DataSet_MachineLearning/Visual/ratings.csv')
ratings = ratings[['userId', 'movieId', 'rating']]
rating_movies = pd.merge(ratings, movies, on='movieId')
ratings_matrix = rating_movies.pivot_table('rating', index='userId', columns='title')
ratings_matrix = ratings_matrix.fillna(0)
ratings_matrix_T = ratings_matrix.transpose()
# (1). 아이템 - 사용자 상호작용 행렬로 변환 = 두 CSV 파일 병합 후 컬럼명 및 결측값 전처리, transpose()를 통해 변환
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
item_sim = cosine_similarity(ratings_matrix_T, ratings_matrix_T)
item_sim_df = pd.DataFrame(data=item_sim, index=ratings_matrix.columns, columns=ratings_matrix.columns)
# (2). 아이템 - 아이템 간 코사인 유사도 계산하여 유사도 행렬로 저장
def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)
# 사용자가 평점을 부여한 영화에 대해서만 예측 성능 평가 MSE 를 구할 수 있음
def predict_rating_topsim(ratings_arr, item_sim_arr, n=20):
    pred = np.zeros(ratings_arr.shape)
    for col in range(ratings_arr.shape[1]):
        top_n_items = np.argsort(item_sim_arr[:, col])[:-n-1:-1]
        for row in range(ratings_arr.shape[0]):
            pred[row, col] = item_sim_arr[col, :][top_n_items].dot(ratings_arr[row, :][top_n_items].T)
            pred[row, col] /= np.sum(np.abs(item_sim_arr[col, :][top_n_items]))
    return pred
ratings_pred = predict_rating_topsim(ratings_matrix.values , item_sim_df.values, n=20)
ratings_pred_matrix = pd.DataFrame(data=ratings_pred, index= ratings_matrix.index, columns = ratings_matrix.columns)
user_rating_id = ratings_matrix.loc[9, :]
def get_unseen_movies(ratings_matrix, userId):
    user_rating = ratings_matrix.loc[userId, :]
    already_seen = user_rating[user_rating > 0].index.tolist()
    movies_list = ratings_matrix.columns.tolist()
    unseen_list = [movie for movie in movies_list if movie not in already_seen]
    return unseen_list
def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n=10):
    recomm_movies = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]
    return recomm_movies
unseen_list = get_unseen_movies(ratings_matrix, 9)
recomm_movies = recomm_movie_by_userid(ratings_pred_matrix, 9, unseen_list, top_n=10)
recomm_movies = pd.DataFrame(data=recomm_movies.values, index=recomm_movies.index, columns=['pred_score'])
# (3). 가중 평점 계산 = 해당 영화의 평점 * 해당 영화의 기준 영화에 대한 유사도 / 전체 유사도 합
# 각 사용자에 대한 가중 평점 행렬 계산 후 계산된 가중 평점을 바탕으로 시청하지 않은 추천 영화 행렬 저장

# 컨텐츠 기반 필터링 = 사용자의 선호 프로파일을 바탕으로 추천 아이템 그룹 식별
# 아이템의 특성을 고려, 사용자 평가기록에 의존 하지 않음 = 아이템 콜드스타트 문제 해결, Sparse 데이터 및 확장성 문제 완화
# 사용자 콜드스타트 문제 해결 불가, 아이템 데이터의 품질에 의존적, 사용자 취향 확장 어려움(다양성, 탐색성 부족)
Movies = pd.read_csv('./DataSet_MachineLearning/Visual/tmdb_5000_movies.csv')
Movies_df = Movies[['id', 'title', 'genres', 'vote_average', 'vote_count', 'popularity', 'keywords', 'overview']].copy()
from ast import literal_eval
Movies_df['genres'] = Movies_df['genres'].apply(literal_eval)
Movies_df['keywords'] = Movies_df['keywords'].apply(literal_eval)
Movies_df['genres'] = Movies_df['genres'].apply(lambda x: [ y['name'] for y in x])
Movies_df['keywords'] = Movies_df['keywords'].apply(lambda x: [ y['name'] for y in x])
# 장르 및 키워드 컬럼을 처리하기 유리하도록 딕셔너리로 변환 후 name 컬럼 문자열만 추출
from sklearn.feature_extraction.text import CountVectorizer
Movies_df['genres_literal'] = Movies_df['genres'].apply(lambda x : ' '.join(x))
count_vect = CountVectorizer(min_df=1, ngram_range=(1,2))
genre_mat = count_vect.fit_transform(Movies_df['genres_literal'])
# (1). 텍스트 정보들을 피쳐 벡터화 = 장르 단어를 공백을 붙여 결합, 장르 단어 빈도 벡터화
# 단어 빈도 벡터 = 전체 문장에서 등장하는 단어들에 대해 해당 문장에서 등장여부를 0과 1로 표기
# 문장1 [1, 0, 1, 1, 1, 0....] / 문장2 [0, 1, 1, 1, 0, 0....] / 문장3.....
# 여기서는 한 영화에 대한 한 장르 단어의 빈도를 표기
from sklearn.metrics.pairwise import cosine_similarity
genre_sim = cosine_similarity(genre_mat, genre_mat)
genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1]
def find_sim_movie(df, sorted_ind, title_name, top_n=10):
    title_movie = df[df['title'] == title_name]
    title_index = title_movie.index.values
    similar_indexes = sorted_ind[title_index, : top_n]
    print(similar_indexes)
    similar_indexes = similar_indexes.reshape(-1)
    return df.iloc[similar_indexes]
similar_movies = find_sim_movie(Movies_df, genre_sim_sorted_ind, 'The Godfather',10)
# (2). 콘텐츠별 유사도 계산 = 영화간 장르 단어 빈도의 유사성을 계산하여 유사성 행렬로 저장
# 영화의 인덱스를 행렬의 유사도 값을 기준으로 내림차순 정렬하여 배열로 저장
# 특정 영화 기준 유사도가 가장 큰 영화들의 인덱스를 배열에서 참조하여 행렬에서 영화명 검색
# 코사인 유사도 = 두 벡터간 방향의 유사성, 피어슨 상관계수 = 두 정규화된 벡터간 방향의 유사성, MSD = 두 벡터의 거리
percentile = 0.6
C = Movies_df['vote_average'].mean()
M = Movies_df['vote_count'].quantile(percentile)
def weighted_vote_average(record):
    V = record['vote_count']
    R = record['vote_average']
    return ((V / (V + M)) * R) + ((M / (M + V)) * C)
Movies_df['weighted_vote'] = Movies_df.apply(weighted_vote_average, axis=1)
print(Movies_df['weighted_vote'])
# (3). 콘텐츠별 가중 평균 계산 = (V / V + M) * R + (M / V + M) * C
# V = 해당 영화의 평점 투표수, M = 평점이 부여되기 위한 최소 투표수, R = 해당 영화의 평균 평점, C = 전체 영화의 평균 평점

# 잠재요인 협업필터링
# Sparse한 사용자 - 아이템 행렬 => 사용자 - 잠재요인 행렬 * 잠재요인 - 아이템 행렬 => Dense한 사용자 - 아이템 행렬
# M x N 사용자 - 아이템 행렬에서 잠재요인을 K개라 하면, M x K 사용자 - 잠재요인 행렬과 K x N 잠재요인 - 아이템 행렬
# 계산 후 내적하면 M x N의 Dense한 사용자 - 아이템 행렬 얻을 수 있음
# 행렬 분해는 경사하강법 이용 = 두 행렬을 임의 행렬로 설정 후 실제 행렬과 예측 행렬의 오류 감소하는 방향으로 행렬 업데이트
R = np.array([[4, np.NaN, np.NaN, 2, np.NaN ],
              [np.NaN, 5, np.NaN, 3, 1 ],
              [np.NaN, np.NaN, 3, 4, 4 ],
              [5, 2, 1, 2, np.NaN ]])
num_users, num_items = R.shape
K = 3
P = np.random.normal(scale=1./K, size=(num_users, K))
Q = np.random.normal(scale=1./K, size=(num_items, K))
# (1). 잠재요인 행렬 P와 Q를 임의의 행렬로 초기화
def get_rmse(R, P, Q, non_zeros):
    error = 0
    full_pred_matrix = np.dot(P, Q.T)

    # 실제 R 행렬에서 널이 아닌 값의 위치 인덱스 추출하여 실제 R 행렬과 예측 행렬의 RMSE 추출
    x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
    y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
    R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]
    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]

    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
    rmse = np.sqrt(mse)
    return rmse
# (2). 예측행렬과 실제행렬간의 비용함수 계산
non_zeros = [ (i, j, R[i,j]) for i in range(num_users) for j in range(num_items) if R[i,j] > 0 ]
steps=1000
learning_rate=0.01
r_lambda=0.01
for step in range(steps):
    for i, j, r in non_zeros:
        eij = r - np.dot(P[i, :], Q[j, :].T)
        P[i,:] = P[i,:] + learning_rate*(eij * Q[j, :] - r_lambda*P[i,:])
        Q[j,:] = Q[j,:] + learning_rate*(eij * P[i, :] - r_lambda*Q[j,:])
    rmse = get_rmse(R, P, Q, non_zeros)
    if (step % 50) == 0 :
        print("### iteration step : ", step," rmse : ", rmse)
# (3). 비용함수 감소하는 방향으로 경사하강법 실시
def matrix_factorization(R, K, steps=200, learning_rate=0.01, r_lambda=0.01):
    num_users, num_items = R.shape
    np.random.seed(1)
    P = np.random.normal(scale=1. / K, size=(num_users, K))
    Q = np.random.normal(scale=1. / K, size=(num_items, K))

    break_count = 0

    non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i, j] > 0]

    for step in range(steps):
        for i, j, r in non_zeros:
            eij = r - np.dot(P[i, :], Q[j, :].T)
            P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
            Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])

        rmse = get_rmse(R, P, Q, non_zeros)
        if (step % 10) == 0:
            print("### iteration step : ", step, " rmse : ", rmse)

    return P, Q
# (1) (2) (3)을 하나의 함수로 만든 것
movies2 = pd.read_csv('./DataSet_MachineLearning/Visual/movies.csv')
ratings2 = pd.read_csv('./DataSet_MachineLearning/Visual/ratings.csv')
ratings2 = ratings2[['userId', 'movieId', 'rating']].copy()
rating_movies2 = pd.merge(ratings2, movies2, on='movieId')
ratings_matrix2 = rating_movies2.pivot_table('rating', index='userId', columns='title')
P, Q = matrix_factorization(ratings_matrix2.values, K=50, steps=200, learning_rate=0.01, r_lambda = 0.01)
# 경사 하강법의 결과값 잠재요인 행렬 저장
pred_matrix = np.dot(P, Q.T)
ratings_pred_matrix = pd.DataFrame(data=pred_matrix, index= ratings_matrix2.index, columns = ratings_matrix2.columns)
def get_unseen_movies(ratings_matrix, userId):
    user_rating = ratings_matrix.loc[userId, :]
    already_seen = user_rating[user_rating > 0].index.tolist()
    movies_list = ratings_matrix.columns.tolist()
    unseen_list = [movie for movie in movies_list if movie not in already_seen]
    return unseen_list
def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n=10):
    recomm_movies = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]
    return recomm_movies
unseen_list = get_unseen_movies(ratings_matrix2, 9)
recomm_movies = recomm_movie_by_userid(ratings_pred_matrix, 9, unseen_list, top_n=10)
recomm_movies = pd.DataFrame(data=recomm_movies.values,index=recomm_movies.index,columns=['pred_score'])
print(recomm_movies)
# 잠재요인 행렬 내적한 Dense 사용자 - 아이템 행렬 기반으로 미시청 영화 추천목록 출력

# Surprise 패키지로 구현하기
import surprise
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
dataset = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(dataset, test_size=.25, random_state=0)
# 데이터 로드 후 테스트데이터와 훈련데이터 분할
algorythm = SVD(n_factors=10, random_state=10)
algorythm.fit(trainset)
predictions = algorythm.test(testset)
# test() 함수는 리스트로 평점 예측 데이터 전체 반환
uid = str(196)
iid = str(302)
pred = algorythm.predict(uid, iid)
# predict() 함수는 한 가지 케이스만 반환
accuracy.rmse(predictions)
# 예측 정확도 계산
ratings_surprise = pd.read_csv('./DataSet_MachineLearning/Visual/ratings.csv')
ratings_surprise.to_csv('./DataSet_MachineLearning/Visual/ratings_surprise.csv', index=False, header=False)
from surprise import Reader
reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))
data = Dataset.load_from_file('./DataSet_MachineLearning/Visual/ratings_surprise.csv', reader = reader)
# , 로 컬럼 구분하며 각각 user, item, rating, timestamp 이며 rating의 범위는 0.5 ~ 5라는 의미
# Surprise는 Low-Level Data만 처리할 수 있으며 user, item, rating 순서를 반드시 지켜야함
data_bydf = Dataset.load_from_df(ratings_surprise[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data_bydf, test_size=.25, random_state=0)
algo = SVD(n_factors=50, random_state=0)
algo.fit(trainset)
predictions = algo.test( testset )
accuracy.rmse(predictions)
# Pandas Dataframe에서 불러오는 것도 가능
# Surprise 패키지 추천 알고리즘
# SVD : 행렬분해 이용한 잠재요인 협업 필터링, 하이퍼파라미터 : n_factors = 잠재요인 K 개수 / n_epochs = SGD 반복회수 / biased = 사용자 편향 적용 여부
# KNNBasic : 최근접 협업 필터링, BaselineOnly : 사용자 성향 반영 협업 필터링, 전체 사용자 평균 + 사용자 편향점수 + 아이템 편향점수
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
param_grid = {'n_epochs': [20, 40, 60], 'n_factors': [50, 100, 200] }
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])
# 교차검증 및 GridSearchCV도 가능
from surprise.dataset import DatasetAutoFolds
data_folds = DatasetAutoFolds(ratings_file='./DataSet_MachineLearning/Visual/ratings_surprise.csv', reader=reader)
trainset = data_folds.build_full_trainset()
# Surprise에서 SVD의 경우 테스트모델 분리없이 전체 데이터를 학습시키고자 하면 DatasetAutoFolds통해 데이터셋 불러와야함
algo = SVD(n_epochs=20, n_factors=50, random_state=0)
algo.fit(trainset)
movies_surprise = pd.read_csv('./DataSet_MachineLearning/Visual/movies.csv')
ratings_surprise = pd.read_csv('./DataSet_MachineLearning/Visual/ratings.csv')
def get_unseen_surprise(ratings, movies, userId):
    seen_movies = ratings[ratings['userId'] == userId]['movieId'].tolist()
    total_movies = movies['movieId'].tolist()
    unseen_movies = [movie for movie in total_movies if movie not in seen_movies]
    print('평점 매긴 영화수:', len(seen_movies), '추천대상 영화수:', len(unseen_movies), \
          '전체 영화수:', len(total_movies))
    return unseen_movies
def recomm_movie_by_surprise(algo, userId, unseen_movies, movies, top_n=10):
    predictions = [algo.predict(str(userId), str(movieId)) for movieId in unseen_movies]

    # predictions list 객체는 surprise의 Predictions 객체를 원소로 가지고 있음.
    # [Prediction(uid='9', iid='1', est=3.69), Prediction(uid='9', iid='2', est=2.98),,,,]
    # 이를 est 값으로 정렬하기 위해서 아래의 sortkey_est 함수를 정의함.
    # sortkey_est 함수는 list 객체의 sort() 함수의 키 값으로 사용되어 정렬 수행.
    def sortkey_est(pred):
        return pred.est

    # sortkey_est( ) 반환값의 내림 차순으로 정렬 수행하고 top_n개의 최상위 값 추출.
    predictions.sort(key=sortkey_est, reverse=True)
    top_predictions = predictions[:top_n]

    # top_n으로 추출된 영화의 정보 추출. 영화 아이디, 추천 예상 평점, 제목 추출
    top_movie_ids = [int(pred.iid) for pred in top_predictions]
    top_movie_rating = [pred.est for pred in top_predictions]
    top_movie_titles = movies[movies.movieId.isin(top_movie_ids)]['title']
    top_movie_preds = [(id, title, rating) for id, title, rating in
                       zip(top_movie_ids, top_movie_titles, top_movie_rating)]
    return top_movie_preds
unseen_movies_9 = get_unseen_surprise(ratings_surprise, movies_surprise, 9)
top_movie_preds = recomm_movie_by_surprise(algo, 9, unseen_movies_9, movies_surprise, top_n=10)
print('##### Top-10 추천 영화 리스트 #####')
for top_movie in top_movie_preds:
    print(top_movie[1], ":", top_movie[2])
# Surprise 기반 영화 추천 시스템
