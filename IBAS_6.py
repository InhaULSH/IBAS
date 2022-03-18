#클래스
#멤버 변수와 멤버 함수의 집합이야
#변수를 여러개 선언하기 - 단순무식
DalPangE1_HP = 50
DalPangE2_HP = 50
DalPangE3_HP = 50
DalPangE4_HP = 50
DalPangE5_HP = 50
DalPangE6_HP = 50
print("and so on....")
#자료구조로 묶어보기 - 좀 더 낫지만 여전히 단순무식
DalPangE_list_HP = [50]
DalPangE_list_MP = [30]
Unit_list_HP = [100]
Unit_list_MP = [120]
print("and so on....")
#구조체 - 멤버 변수의 집합, 단순무식에서 벗어났지만 구조체 안에서 어떤 작업을 할 수가 없음
# struct DalPangE() {
#   int HP = 50;
#   int MP = 30;
#   int AD = 10;
# }
# DalPangE1 = DalPangE();
# DalPangE2 = DalPangE();
# 달팽이의 기본 설계도를 가지로 달팽이 두마리를 만들었어
# 보통 작업은 함수로 구현할 수 있지만 일일이 작업을 구현하는 단순무식한 방법을 써야돼
#클래스 - 멤버변수와 멤버함수의 집합, 클래스 안에서 개인화된 작업까지 할 수 있어!
class DalPangE :
    HP = int() #멤버 변수 - 클래스 설계도 내부의 변수
    MP = int()
    AD = int()
    DF = int()
    def move(self): #멤버 함수 - 클래스 설계도 내부의 함수
        return 0
#클래스 & 상속 - 클래스를 일일히 선언해줄 필요없이 공통적인 멤버변수와 멤버 함수들을 물려줄 수 있어!
class Field_Unit : #부모 클래스 - 클래스간 공통적인 요소를 가진 클래스야
    HP = int()
    MP = int()
    AD = int()
    DF = int()
    def move(self):
        return 0
class DalPangE_R(Field_Unit) : #자식 클래스 - 부모 클래스로부터 요소를 물려받는 클래스야
    def DPE_crash(self):
        return 0
#상속을 이용하면 자식 클래스가 공통 기능을 가지면서도 자식 클래스의 고유의 기능을 추가할 수 있음!
#정리하면  클래스 - 멤버변수와 멤버함수의 집합
#         멤버 변수 - 클래스 설계도 내부의 변수
#         멤버 함수(메소드) - 클래스 설계도 내부의 함수
#         인스턴스 - 클래스 설계도로 인해 생성된 변수
#클래스를 선언하기
class class_name :
    def __init__(self) : #파이썬 내부에서 지정된 멤버함수 : 생성자!
        self.HP = int() #멤버 변수들
        self.MP = int()
        self.Name = str()
    def print_info(self) : #멤버 함수
        print("HP : ",self.HP,"MP : ",self.MP,"Name : ",self.Name)
    def move_to_user(self):
        print("유닛이 유저에게 다가갑니다")
Unit = class_name() #인스턴스
Unit.print_info()
#생성자(__init__) - 인스턴스가 만들어 질때 실행되는 함수, 인스턴스를 반환하고 멤버변수를 초기화하는 역할을 수행해!
#이때 self는? - 인스턴스 변수 자신을 가리키는 파이썬 클래스 내장변수 즉, Unit과 같음
#대중적으로 사용되는 생성자 활용 패턴
class Enemy_Unit :
    def __init__(self, HP, MP, Name):
        self.HP = HP
        self.MP = MP
        self.Name = Name
Enemy = Enemy_Unit(100,120,'적1')
#이렇게 인스턴스를 만들때(생성자를 실행할때) 멤버변수의 값을 받아와서 초기화 시킬 수 있어!
#이외에도 멤버변수를 생성자 함수 내부에서 값을 지정하거나, 클래스 외부에서 값을 지정해 줄 수 있어
#생성자를 이용해 사칙연산 계산기 만들기
class FourRes_Calculater :
    def __init__(self,Var1,Var2) : #Var1,Var2는 매개변수
        self.Var1 = Var1 #self.Var1,self.Var2는 멤버변수 즉, 둘은 이름만 비슷한 다른 변수!
        self.Var2 = Var2
    def print_Res(self):
        print(self.Var1,"+",self.Var2,"=",self.Var1+self.Var2,"\n")
        print(self.Var1,"-",self.Var2,"=",self.Var1-self.Var2,"\n")
        print(self.Var1,"*",self.Var2,"=",self.Var1*self.Var2,"\n")
        print(self.Var1,"/",self.Var2,"=",self.Var1/self.Var2,"\n")
FourRes_Calculater(3,4)
FourRes_Calculater.print_Res()
#상속의 자세한 설명
#부모 클래스의 멤버변수와 멤버함수를 가져와서 자식클래스에서 그대로 사용하는 거야
class DPE(class_name) :
    def __init__(self,Color):
        self.Color = Color
    def move_to_user(self):
        print("달팽이가 유저에게 다가갑니다")
#class_name이라는 부모 클래스,자식 클래스 DPE가 존재해
#이때 부모 클래스에서 상속받은 멤버함수 중 같은 이름의 함수를 새로운 내용으로 덮어쓰기할 수 있어 - 이걸 오버라이딩이라고 해
#다중상속이라는 것도 있어 부모 클래스를 여러개 가지도록 상속하는 건데, 이렇게 하는 건 굉장히 위험해
#부모 클래스의 멤버 변수나 멤버 함수 이름이 겹칠 경우 어떤 걸 상속받아야 될지 모르기 때문이야
#포함 - 클래스의 멤버변수로 다른 클래스의 인스턴스 변수를 선언하는 거야
class Sword :
    def __init__(self):
        self.AD = 100
class Character :
    def __init__(self,HP,MP,Weapon):
        self.HP = HP
        self.MP = MP
        self.Weapon = Weapon
    def Attack(self):
        print("캐릭터가 ",self.AD,"의 공격력으로 공격합니다")
Char1 = Character(1000,200,Sword()) #Sword()는 Sword 클래스의 인스턴스
Char1.Attack()

