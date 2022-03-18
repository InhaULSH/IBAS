# 세부 데이터 수집
# 구글 API 데이터 뿐만 아니라 브라우저를 직접 제어해서 더욱 세부적인 데이터를 가져올 수 있음
# 우선 브라우저 종류와 버전에 맞는 웹드라이버를 프로젝트 폴더에 설치해야함

# 한 동영상의 전체 댓글 데이터 수집하기
from bs4 import BeautifulSoup
from selenium import webdriver
import time

url = "https://www.youtube.com/watch?v=3lwTql6YlSE" # 분석하려는 유투브 동영상 주소
driver = webdriver.Edge('msedgedriver')
driver.maximize_window()
driver.get(url)

for i in range(5) :
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(5) # 자바 스크립트로 화면 스크롤하기

LimousineSoup = BeautifulSoup(driver.page_source, 'html.parser')
driver.close()

# HTML 태그의 이름과 아이디를 이용해서 페이지의 특정 소스만 긁어올 수 있음
LimousineComments = LimousineSoup.select("yt-formatted-string#content-text") # 특정 태그의 내용을 긁어오기
LimousineCommentsAuthors = LimousineSoup.select("div#header-author a#author-text") # 특정 태그의 하위 태그 중 특정 태그의 내용을 긁어오기
LimousineData = list()
if (len(LimousineComments) == len(LimousineCommentsAuthors)) :
    for i in range(len(LimousineComments)) :
        TemporaryList = list()
        TemporaryList.append(LimousineComments[i])
        TemporaryList.append(LimousineCommentsAuthors[i])
        LimousineData.append(TemporaryList)
for i in range(len(LimousineData)) :
    print(LimousineData[i][0].text.replace("\n", " ").replace("\t", "").strip(), " by ",LimousineData[i][1].text.replace("\n", " ").replace("\t", "").strip())
