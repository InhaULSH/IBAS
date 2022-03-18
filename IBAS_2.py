# None 변수 -> 값이 존재하지 않는 변수야, 오류방지용으로 주로 씀 or 미리 선언해두는 용도로 씀
# X = None, print(X) = None

# 자료구조 -> 자료를 저장하는 형태야
# 1. 리스트(가장 중요!!)
# 자료를 배열한 형태로 저장함, [ 1 / 2 / 3 / 4 / 5 ]와 같은 형태
# 'Index'를 통해 리스트의 값에 접근 -> 리스트 속의 가상 주소, [0]부터 시작해서 [1], [2]...처럼 주소가 부여돼
std_list1 = list() #빈 리스트의 선언
std_list2 = [12,34,56,78,90] #자료가 들어있는 리스트의 선언
std_list3 = ["가","나",True] #문자나 bool도 들어갈 수 있어
print(std_list2) #리스트 속 값을 출력하기
print("리스트형 변수인 std_list2 그 자체를 출력") #std_list2라는 리스트형 변수를 까봤더니 리스트 자체가 나왔어
print(std_list2[0])
print(std_list2[1])
print(std_list2[2])
print(std_list2[3])
print(std_list2[4])
print("리스트형 변수의 요소를 출력") #std_list2를 까봤더니 0~4번지가 있어서 그쪽으로 가봤더니 정수변수가 나왔어 -> 참조형 변수
#이렇게 찾아가는 과정이 'Indexing', 변수뒤 대괄호에 들어간 주소를 'Index'
#1-1. 이중리스트
double_list = [1,2,[3,4,5],6]
print(double_list[2])
print(double_list[2][1])
#리스트 안에 리스트를 품은 변수!
#1-2. 리스트의 수정
double_list.append(78) #리스트에 요소 추가하기, ()안의 값을 가장 뒤에 요소로 추가해줌
print(double_list)
double_list[2].append(67) #이중리스트에 요소 추가하기
print(double_list)
double_list.pop(0) #리스트 요소를 없애기, 리스트.pop(인덱스)로 하면 됨!
print(double_list)
double_list[3] = 7 #리스트 요소 수정하기,리스트[인덱스] = 값으로 하면 됨!
print(double_list)
double_list.append([1,2,3]) #리스트에 이중리스트 추가하기
print(double_list)
double_list.clear() #리스트를 비우기
print(double_list)
#1-3. 슬라이싱
double_list = [1,2,[3,4,5],6]
print(double_list[0:2]) #부분만 잘라서 호출, 변수의 요소를 다루기위해 고안된 개념이야
# 0번 인덱스부터 2번 인덱스전까지(Until 2) 출력하라는 뜻
#1-4. String변수의 인덱싱
Hello_str = "Hello"
print(Hello_str[0])
print(Hello_str[1])
print(Hello_str[0:4]) #String변수는 튜플의 성질을 가짐! -> 수정 불가, 인덱싱 가능

# 2. 튜플(별로 안중요함)
# 수정할 수 없는 리스트, 리스트랑 다르게 상수야
std_tuple0 = tuple()
#std_tuple.append(0) #이렇게 못함
std_tuple = (1,2,3,4,5) #이렇게 선언해줘야해!
print(std_tuple)
print(std_tuple[2]) #호출은 리스트랑 똑같이 가능!

# 3. 딕셔너리(어느정도 중요함)
# 사전 자료형, 사전처럼 'Key'와 'Value'가 짝지어진 쌍을 변수로 저장하는 자료형 -> 자동차 : Car, 사람 : Human ....
# 'Key'와 'Value'가 짝지어진 쌍 = 엔트리, 딕셔너리를 엔트리를 저장
std_dict0 = dict()
std_dict = {} #빈 딕셔너리 추가
std_dict['Red'] = '빨강' #Key를 인덱스로 삼아 접근할 수 있음, 수정뿐만 아니라 추가까지 할 수있음!, 딕셔너리[Key] = Value로 할 수 있음
print(std_dict['Red']) #딕셔너리 Key를 호출하기
std_dict['Red'] = '적색'
print(std_dict['Red']) #Value변경, Key값은 변경불가!

# 4. 집합(별로 안중요함)
# 딕셔너리와 비슷하지만 인덱싱이 안됨, 순서가 없거든
num_set = {1,3,5,7,9}
print(num_set)
num_set = {-1,-1,1,3,5,11,9,7}
print(num_set) #중복을 제거해줌!
#합,차,여집합은 구현가능!