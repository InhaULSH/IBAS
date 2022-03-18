# 데이터 수집
# -> API를 통해서
# -> GCP의 권한을 받아보기 -> Request 라이브러리 사용해서 API를 사용하기 -> 데이터(JSON)을 딕셔너리로 사용하기 -> 내 의도에 맞추어 사용하기(정보의 표시, 리스트로 정리)

# GCP 권한 받아보기
# 구글 클라우드 플랫폼 - 프로젝트 - API 개요로 이동 - API 및 서비스 사용 설정 - Youube Data API v3 - 사용
# Youube Data API v3 - 사용자 인증 정보 - 사용자 인증정보 만들기 - API 키

# Request 라이브러리 사용해서 API를 사용하기
import requests
import json
url = "https://www.googleapis.com/youtube/v3/videos"  # 데이터를 먼저 전달받음
parameter = {
    "key" : "AIzaSyA54XC3O3ghWz4_KUo6gGYlJBNJFGGPJQY",  # 인증정보 키가 필요함
    "id" : "BJNHoRrZw7k",  # 영상의 ID가 필요함, URL에서 v= 으로 되어있는 부분임
    "part" : ["snippet", "statistics"]  # 어떤 정보를 불러올지가 필요함, 설명서에서 확인 가능함
}
response = requests.get(url, params = parameter)  # request를 거친 데이터가 저장됨
print(response, type(response))  # 200은 정상이란 뜻임 400은 키값이 잘못 되었다는 뜻, requests.models.Response 타입으로 확인됨

# 데이터(JSON)을 딕셔너리로 사용하기
data = response.json()
print(data, type(data))  # 영상의 정보들이 담겨 있음, 딕셔너리 타입으로 확인됨
# 이 딕셔너리의 구조는 설명서에서 확인 가능함
print(data["items"][0]["snippet"]["title"])
print("조회수 : ", data["items"][0]["statistics"]["viewCount"])  # 딕셔너리와 리스트가 중첩된 구조임, 따라서 딕셔너리와 리스트 인덱싱을 통해 부분적으로 정보 추출가능

# 내 의도에 맞추어 사용하기 - 인기 동영상 10개에서 정보 추출해보기
import requests
import json
import pprint # 예쁘게 출력하기 위해 사용

class YoutubeCrawl:
    def __init__(self):
        self.url = "https://www.googleapis.com/youtube/v3/"
        self.key = "AIzaSyA54XC3O3ghWz4_KUo6gGYlJBNJFGGPJQY"

    def getPopularVideos(self):
        parameter = {
            "key": self.key,
            "part": ["snippet", "statistics"],
            "chart": "mostPopular",
            "regionCode": "KR",
            "maxResults": 10
        }
        response = requests.get(self.url + "videos", params=parameter)
        data = response.json()["items"]
        video_list = list()
        for i, item in enumerate(data, start=1):
            dict_for = dict()
            dict_for["rank"] = i
            dict_for["id"] = item["id"]
            dict_for["channelTitle"] = item["snippet"]["channelTitle"]
            dict_for["title"] = item["snippet"]["title"]
            dict_for["viewCount"] = item["statistics"]["viewCount"]
            video_list.append(dict_for)
        print(video_list)
        return video_list

    def getVideoComments(self, Id):
        parameter_c = {
            "key" : self.key,
            "part" : "snippet",
            "videoId" : Id,
            "maxResults" : 100,
            "order" : "time",
            "textFormat" : "plainText"
        }
        response_c = requests.get(self.url + "commentThreads", params=parameter_c)
        data_c = response_c.json()["items"]
        comment_list = list()
        for item in data_c :
            comment_list.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"].replace("\n","").strip())
        return comment_list
    # 크롤링 클래스, 비디오와 댓글 정보 크롤링하는 기능을 클래스로 구현

Crawler = YoutubeCrawl()
PP = pprint.PrettyPrinter(indent=4)
VideoData = Crawler.getPopularVideos()
for i in range(len(VideoData)):
    VideoData[i]["comments"] = Crawler.getVideoComments(VideoData[i]["id"]) # 댓글 막아놓으면 오류남
PP.pprint(VideoData)
# 메인 함수, 클래스를 이용해 크롤링 데이터 출력