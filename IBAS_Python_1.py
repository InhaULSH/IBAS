#주석은 두가지 종류가 있음
#1. 한줄 주석 - 문장 앞에 '#'으로
#2. 여러줄 주석 - '''주석 내용'''
#여러줄 주석은 잘 안써

#변수
#-> 값이나 주소가 저장되고 저장된 값을 바꿀 수 있어(x = 10)
#상수는 저장된 값을 바꿀 수 없어(_x = 10)
#기본형 변수는 값을 담은 상자, 참조형 변수는 값이 있는 주소를 알려주는 표지판
#변수의 선언
#No = int() #뭔지는 모르겠지만 정수가 들어갈 변수 No가 있어
#Str = str() #뭔지는 모르겠지만 문자가 들어갈 변수 Str이 있어
#변수의 초기화
#변수에 값을 넣거나 변경하면돼 (x = int() -> x = 123)
#파이썬은 변수 선언이랑 초기화를 같이 해(x = 10 , 정수 변수를 선언하면서 10으로 초기화)
#변수의 호출
#print(여기에는 값이 들어가야해), 호출은 명령어에서 변수를 쓰기위해서 변수를 내용물에 따라 값으로 바꿔주는 작업

#자료형
#1.기본 자료형 -> like 성적표의 과목별 성적, 쪼갤 수 없는 자료형, 정수형(int)->소수점 제외 정수, 실수형(float)->소수점 포함 실수, 문자형(String)->문자열, boolean형-> 참거짓을 저장
#2.참조 자료형 -> like 성적표, 쪼갤 수 있는 자료형, 2강에서 배울거임

#연산자
#1. 산술연산자 -> +(합),-(차),*(곱),/(몫),%(나머지)
#   Str형의 합과 곱 -> +(앞뒤 문자열을 붙여서 출력),*(곱한 숫자만큼 문자열을 반복출력)
#2. 대입연산자 -> =(변수에다가 값을 넣어줌)
#3. 대소비교연산자 -> ==(같다),!=(같지 않다),<(미만),>(초과),<=(이하),>=(이상) --> 숫자가 아닌 참거짓으로 출력
#4. 논리연산자 -> and(그리고) or(또는) -->대소비교연산자와 bool형 자료형과 같이 등장!
#   True and True = True -> True, False and bool형 = True -> False 등
