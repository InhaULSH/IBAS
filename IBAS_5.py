#파이썬 내장함수
#파이썬을 만든 사람이 만들어 놓은 함수, 사람들이 많이 쓰는 함수들이야
#print 함수
print("매개변수")
#매개변수 입력값을 콘솔에 뿌려주고 아무것도 반환하지 않아 / 입력은 있고 반환은 없는 함수
#input 함수
input()
#사용자의 키보드로 부터 매개변수를 입력받는 함수야, 입력받고나서 입력받은 값을 str형태로 반환해 / 입력과 반환 둘다 있는 함수
input("????")
#괄호안에 값을 넣으면 입력을 받으면서 메시지를 출력할 수 있어 / print(메세지) + input()
Var1 = input("변수1 : ")
#당연히 변수에 넣는것도 가능! 당연히 이때 변수에 저장된 건 문자열이야
Var2 = int(input())
#형변환 함수를 통해 특정 자료형으로 인풋을 받을 수 있어
#리스트 관련 함수
list_name = []
len(list_name)
#리스트를 입력 받아서 리스트의 길이를 반환하는 함수야
list_name.append(3)
#리스트 마지막에 매개변수로 입력된 값을 추가하고 아무것도 반환하지 않는 함수야
list_name = [1,2,3,4,5]
list_name.sort()
#정수 기준으로 리스트를 오름차순 정렬하는 함수야
list_name.sum()
#리스트나 튜플 받아서 합을 출력하는 함수야
list_name.max()
list_name.min()
#리스트나 튜플 받아서 최대/최소를 출력하는 함수야
def args_sum(*args) :
    sum = 0
    for no in args :
        sum += no
    return args
args_sum(1,2,3,4,5,6,7,8,9,10)
#*은 매개변수를 개수상관없이 받을 수 있는 함수야
#*로 여러 개의 매개변수를 받으면 튜플로 변환해서 함수에 매개변수를 입력해줘
#print같은 함수도 *을 통해 입력을 받는다고 생각하면되
def kwargs_hello(**kwagrs) :
    if kwagrs['say'] :
        print(kwagrs['cont'])
kwargs_hello(say = True,cont = 'Hello')
#**는 딕셔너리 형태로 변환해서 함수의 매개변수에 입력해주는 함수야
#들어오는 매개변수의 형태를 제한하고 싶을때 사용해

#최솟값/최댓값 구하는 함수를 만들어 보자
#매개변수 = 숫자 리스트, min = True이면 최솟값 False이면 최댓값
def MM_Calculater(*args,**kwargs) :
    temp = args[0]
    for No in args :
        if kwargs['min']:
            if (temp > No) :
                temp = No
        else :
            if (temp < No) :
                temp = No
    return temp
MM_Calculater(1,2,3,4,5,min = False)
MM_Calculater(1,2,3,4,5,min = True)

#정렬의 알고리즘
# 2 / 3 / 1 / 4 / 7
#최솟값을 찾고 왼쪽으로 넘기고, 다시 나머지 중에 최솟값을 찾아서 아까 찾은 값을 뺀 왼쪽으로 넘기고를 반복하는게 선택정렬이야
#sort함수를 만들때는 새로운 리스트를 만들어서 위의 알고리즘을 활용해서 반복해서 옮기면 돼
