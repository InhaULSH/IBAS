#과제 - 인덱스 찾기
def list_finder(target_list: list, keword) :
    listlong = len(target_list)
    for index in range(listlong):
        if (target_list[index] == keword) :
            return index
list_finder([1,2,2,2,2,3,1,3],2)
#if문에서 None은 어지간하면 쓰지말자, 대신 나올 수 없는 값 따위를 리턴하도록 만들자

#과제 - 성적 평균 출력하기
def studnet_avg(grade_dict: dict) :
    sum = 0
    for key in grade_dict.keys() : #또는 for key in list(grade_dict)
        #여기서 sum을 선언하면 for문이 돌때마다 0으로 초기화돼 여기서 선언하면 안됨!
        sum += grade_dict[key]
    avg = float()
    avg = sum / len(grade_dict.keys())
    return avg
studnet_avg({"국어": 85,"수학": 74,"영어": 91})

#과제 - 별 찍는 함수 만들기
def printstar(times) :
    print("★"*times)
for X in range(5) :
    printstar(5-X)

#반환
def func(inputVar = int(input())) :
    while True :
        if (inputVar == -1) :
            return 0
        else :
            print(inputVar)
#반환하면 함수(def)가 즉시 종료돼 따라서 함수를 원하는 타이밍에 종료시키고 싶을때 사용하면 유용해
#반환하면 함수가 값을 내놓은 다음 값이 아무데도 쓰이지 않고 둥둥 떠있는 상태야 이 상태에서 호출하면 리턴값을 사용할 수 있어
#break는 조건문, 반복문 만 종료되지만 return은 함수 전체를 종료시켜, 특정 코드를 독립적으로 실행시키고 싶을때 사용하면 좋아

#딕셔너리.keys() : 키들을 리스트로 반환
#딕셔너리.values() : 밸류들을 리스트로 반환