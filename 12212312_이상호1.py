#변수의 호출에 대해 설명하기
print("파이썬의 명령어 안에는 값이 들어가야 제대로된 문법이다. 따라서 변수의 호출이란 파이썬 명령어에서 변수를 활용하기 위해 변수를 변수에 담긴 값으로 바꿔주는 과정이다. ")

#나머지 연산자를 통해 홀짝구분법
print("짝수는 일반화하면 2n, 홀수는 일반화 하면 2n+1이라고 할 수 있다. 따라서 '정수 변수%2'가 0이면 짝수이고 1이면 홀수다.")

#대소비교연산자를 통해 정수x가 홀짝인지 밝히기
X = int(input())
print(X%2 != 0)

#동전 계산기 만들기 (money에는 돈의 액수가 들어가고 그에 따라 돈이 몇개가 필요한지 출력하기
money = int(input())
print("5만원",money // 50000,"개")
money = money % 50000
print("만원",money // 10000,"개")
money = money % 10000
print("5천원",money // 5000,"개")
money = money % 5000
print("천원",money // 1000,"개")
money = money % 1000
print("5백원",money // 500,"개")
money = money % 500
print("백원",money // 100,"개")
money = money % 100
print("오십원",money // 50,"개")
money = money % 50
print("십원",money // 10,"개")
money = money % 10
print("남은 잔돈",money,"원")