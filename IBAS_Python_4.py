#함수
#수학에서의 함수랑 정확히 같은 개념이야, 정의역 ---> 치역처럼 입력 ---> 반환하는 구문이야
#수학에서의 함수               vs 컴퓨터공학에서의 함수
#입력과 반환이 모두 존재해야해       입력과 반환이 없을수도 있어
#변수가 1개                       변수가 여러개일 수 있어
#치역이 1개                       반환이 여려개 처럼 보이게 할 수 있어(EX : 리스트 반환값)

#함수의 선언
#def 함수이름(매개변수) :
#   함수내용
#   return 반환값
#수학에서 f(x) = x^2 + 1 로 정의한것 처럼
#여기서 return은 식에 따른 값을 반환하는 친구야
def func1(X) :
    Sum = X
    for i in range(10) :
        Sum += 1
    return Sum
#함수의 호출
#print(함수이름(파라미터))
#함수이름+(파라미터)로 호출할 수 있어
#수학에서 f(1) = 2 로 구한것 처럼
#이때 호출하는 순간 f(1)은 return에 적혀 있는 식대로 연산한 값이 되
print(func1(3))
#입력을 2개 받는 함수
#입력 -> 파라미터(호출), 매개변수(선언)
#def 함수이름(매개변수1,매개변수2) :
#   함수내용
#   return 반환값
def add1(X,Y) :
    return X + Y
#입력이 3개인 함수
#def 함수이름(매개변수1, 매개변수2, 매개변수3) :
#   함수내용
#   return 반환값
def add2(X,Y,Z) :
    return X + Y + Z
#함수를 쓰는 이유?
#함수의 기능을 편리하게 재사용할 수 있음!
#코드를 훨씬 깔끔하게 짤 수 있어
#함수의 형태 1 - 입력은 없고 반환만 있는 함수
def func2() :
    return "Hello"
#함수의 형태 2 - 입력만 있고 반환은 없는 함수
def func3(X) :
    print(X + "회")
#함수의 형태 3 - 입력도 반환도 없는 함수
def func4() :
    print("말하기")
#여러 변수를 매개변수로 받을 수 있는 함수
def Sum_list(num_list) :
    Sum = 0
    for num in num_list :
        Sum += num
    return Sum
print(Sum_list([1,2,3,4,5,6,7,8,9,10]))
#매개 변수의 타입을 미리 알려 줄 수 있어
def Add_list(num_list: list, X) :
    num_list.append(X)
#매개변수와 일반변수는 전혀 달라, 이름이 같더라도!
#지역변수와 전역변수
def region(X) :
    Vari = 1
    return Vari
Vari = None
print(Vari)
#이때 함수내의 Vari는 지역변수, 함수 밖의 Vari는 전역변수
#지역변수는 특정 구문 내부(: 이후)애서만 존재하는 변수야, 함수의 반환 값으로 호출되더라도 함수의 외부에 영향을 주지 못해
#전역변수는 특정 구문 외부에서 존재하는 변수야, 함수 내부에서는 이론상 존재할 수는 있지만
X = 10
def globa(Y) :
    print(X)
    return Y
globa(2)
#같은 이름의 지역변수가 존재하면 함수에서는 전역변수가 무시되
def regioon(X) :
    global Vari
    Vari = 1
    return Vari
#대신 구문안에서 global 키워드를 사용하면 전역변수를 변수로 쓸 수 있어, 이때는 함수 밖에서 함수안의 변수를 건들 수 있어
#그래서 변수는 미리 선언해 두는게 좋아, 지역변수때문에 나는 에러를 방지할 수 있어

#파이썬의 내장함수 -> 함수이름()로 사용!
#내가 만들지 않았고 ()를 붙여서 쓰면 모두 내장함수!
print() #출력
a = 1.2
int(a) #강제로 정수로 형변환 해주는 함수
str(a) #강제로 문자열로 형변환 해주는 함수
list(a) #빈 리스트를 만들거나 리스트 비슷한 것을 강제로 리스트로 형변환 해주는 함수
