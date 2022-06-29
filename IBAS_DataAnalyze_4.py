# 데이터 분석
# 수집한 한국어 데이터를 konlpy를 통해 분석
# WordCloud를 통해 이미지 형태로 한국어 데이터를 시각화

from konlpy.tag import Kkma
from konlpy.tag import Komoran
from collections import Counter
from wordcloud import WordCloud

import csv

Text = '크리스마스에는 코딩이 제맛이지'

Kkoma = Kkma()
print(Kkoma.nouns(Text))
print(Kkoma.pos(Text))
print(Kkoma.morphs(Text))
print(Kkoma.sentences(Text)) # 꼬꼬마를 통한 한국어 데이터 분석, 품사별로 출력하거나 형태소별로 품사를 분석할 수 있음

Komora = Komoran()
print(Komora.nouns(Text))
print(Komora.pos(Text)) # 코모란을 통한 한국어 데이터 분석/, 기능은 꼬꼬마와 유사하나 오타 보정 기능이 있음

TxtFile = open("./IBAS_11 TextFile/Speech.txt", encoding="UTF-8")
Speech = TxtFile.read()
SpeechAnalyze = Komoran()
SpeechNouns = SpeechAnalyze.nouns(Speech)

for index, Noun in enumerate(SpeechNouns):
    if (len(Noun) == 1):
        SpeechNouns.pop(index)
    else :
        continue
# WordCloud에 한 글자짜리 명사는 표시되지 않게 처리, enumerate를 사용하면 리스트의 요소와 인덱스를 동시에 순회함

NounCounter = Counter(SpeechNouns)
NounMostCommon = NounCounter.most_common(10)
for i in NounMostCommon:
    print(i)
TxtFile.close() # 코모란을 통해 기사나 연설에서 명사의 빈도수 표시하기

with open("./IBAS_11 TextFile/ArticleAnalyze.csv", "w+", encoding="utf-8-sig", newline= "") as SaveFile:
    CsvWrite = csv.writer(SaveFile)
    for data in NounMostCommon:
        CsvWrite.writerow(data)
# 분석한 한국어 데이터를 CSV 파일에 저장

Wcloud = WordCloud(font_path="C:/Windows/Fonts/NanumGothic.ttf", background_color="white", width=500, height=500, max_words=20, max_font_size=100)
Wcloud.generate_from_frequencies(dict(NounCounter))
Wcloud.to_file("Wcloud.png")
# WordCloud를 통해 분석한 한국어 데이터를 시각화