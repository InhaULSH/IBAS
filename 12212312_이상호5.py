#과제 - 리스트 선택정렬하는 함수 만들기
# *args를 이용해 수를 받고 kwargs를 이용해 reverse가 False이면 오름차순, True이면 내림차순 정렬하여 리스트 반환하기
def list_sort(*args,**kwargs) :
    input_list = list(args)
    output_list = []
    for X in range(len(input_list)) :
        if kwargs['reverse'] :
            maxVar = max(input_list)
            output_list.append(maxVar)
            input_list.pop(input_list.index(maxVar))
        else :
            minVar = min(input_list)
            output_list.append(minVar)
            input_list.pop(input_list.index(minVar))
    return output_list
list_sort(1,3,2,5,4,reverse = True)
list_sort(1,3,2,5,4,reverse = False) #파이썬 내장함수로 구현하기 / 짜기는 쉽지만 조금 비효율적
print("https://seongjaemoon.github.io/python/2017/12/16/pythonSort") #더 효율적인 방법
#과제 - 점수를 넣어 등급반환하는 함수 만들기
#*args로 점수들을 받아서 90이상은 A. 80이상은 B와 같이 F등급까지로 정한다음 점수들 순서에 따라 등급이 들어가 있는 리스트를 반환할 수 있는 함수를 만들기
def grade_calculater(*args) :
    input_list = list(args)
    output_list = []
    for Y in range(len(input_list)) :
        if (input_list[0] >= 90) :
            output_list.append('A')
            input_list.pop(0)
        elif (90 > input_list[0] >= 80) :
            output_list.append('B')
            input_list.pop(0)
        elif (80 > input_list[0] >= 70) :
            output_list.append('C')
            input_list.pop(0)
        elif (70 > input_list[0] >= 60) :
            output_list.append('D')
            input_list.pop(0)
        else :
            output_list.append('F')
            input_list.pop(0)
    return output_list
grade_calculater(95,43,82,71,68)
