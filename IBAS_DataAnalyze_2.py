# 데이터 저장
# 쉼표로 구분된 CSV 파일에 전송받은 데이터를 저장할 수 있음
# with open("파일의 디렉토리(현재 디렉토리면 생략)/파일이름", "열기 옵션", encoding="인코딩 옵션") as 파일을 호출할 수 있는 이름

import requests
import json
import pprint # 예쁘게 출력하기 위해 사용
import csv

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

with open("./IBAS_9 CSVs/DataSet.csv", "w+", encoding="utf-8-sig", newline="") as DataFile: # CSV 파일 열기(파일 없으면 새로 작성하는 모드)
    csv_Write = csv.writer(DataFile) # CSV 파일 작성 기능을 변수로 저장
    for data in VideoData :
        csv_Write.writerow([data["rank"], data["id"], data["title"], data["channelTitle"], data["viewCount"]]) # CSV 파일에 쉼포로 구분해서 기록

        with open("./IBAS_9 CSVs/" + data["id"] + ".csv", "w+", encoding="utf-8-sig", newline="") as CommentFile :
            csv_comment_Write = csv.writer(CommentFile)
            for comment in data["comments"] :
                csv_comment_Write.writerow([comment])
# CSV 파일에 저장

with open("./IBAS_9 CSVs/DataSet.csv", "r", encoding="utf-8-sig", newline="") as DataFile : # CSV 파일 열기(읽기만 하는 모드)
    csv_Read = csv.reader(DataFile) # CSV 파일 읽기 기능을 변수로 저장
    for data in csv_Read :
        print(data) # CSV 파일의 정보를 출력
# CSV 파일로부터 값 읽음