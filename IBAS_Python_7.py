# 클래스 변수 vs 멤버 변수
class Example() :
    Var = 1 #클래스 변수 - 인스턴스 끼리 공유하는 변수
            #클래스들 끼리 공통적으로 지니는 값이 있을때 주로 써, 파이나 자연상수처럼 고정적인 값이 그 예시야
A = Example()
B = Example()
Example.Var = 2
print(A.Var)
print(B.Var)

# 코딩 패턴 - 메뉴를 출력하기
while True :
    flag = int(input("어떤 작업을 하시겠습니까? \n1.조회 \n2.삽입 \n3.삭제 \n-1.종료 \n"))
    if (flag == 1) :
        print("정보를 조회합니다")
        continue
    elif (flag == 2) :
        print("정보를 추가합니다")
        continue
    elif (flag == 3) :
        print("정보를 삭제합니다")
        continue
    elif (flag == -1) :
        print("프로그램을 종료합니다")
        break
    else :
        print("잘못 입력하셨습니다. 다시 입력하세요!")
        continue

# 코딩 패턴 - 같은 크기의 리스트 동시 접근 with for문
List1 = [1,2,3,4]
List2 = [5,6,7,8] #크기가 같은 두 리스트
for X in range(len(List1)) : #단일 for문 - 4번 실행
    print(List1[X],List2[X])

for X in range(len(List1)) :
    for Y in range(len(List2)) : #이중 for문 - 16번 실행
        if (X == Y) :            #즉, 이중 for문은 가급적 지양해야해! 내부적인 연산으로 치면 상당히 비효율적이야!
            print(List1[X],List2[Y])
        else :
            continue

# 객체 vs 인스턴스
# 생성자가 반환하는 결과를 인스턴스
# 인스턴스를 담고있는 주소를 가진 '변수'를 객체라고 해
# 부모 클래스로 부터 탄생한 아이를 인스턴스라고 한다면
# 그 아이를 담고 있고 주소를 가진 집, 학교를 객체라고 볼 수 있어

# 모듈
# 사용자나 기관에서 자주 사용하는 기능을 재사용하기 위해 만들어둔 코드 뭉치
# 사용하고 싶을때 사용할 수 있도록 만든 코드 부품
# 즉, 파이썬 내장기능이 아닌 외부에서 만들어진 유용한 코드뭉치를 가져오는 것! 'import 모듈이름 as 내 코드에서 쓸 모듈이름'으로 가져올 수 있음!
# import Numpy as Np
# 모듈의 일부 기능만 쓰고 싶을때는 'from 모듈이 있는 주소.모듈 import 기능'과 같이 가져올 수 있어! import 뒤에 있는 게 최종목적지!
# from path.Mod import Add