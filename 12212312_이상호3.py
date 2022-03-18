#과제 - 별찍기
for Star in range(1,10):
    if (Star == 1) or (Star == 9):
        print("☆ "*9)
    elif (Star >= 2) and (Star <= 4):
        print("★ "*(Star-1)+"☆ "+"★ "*(9-2*Star)+"☆ "+"★ "*(Star-1))
    elif (Star == 5):
        print("★ "*4+"☆ "+"★ "*4)
    else:
        print("★ "*(9-Star)+"☆ "+"★ "*(9-2*(10-Star))+"☆ "+"★ "*(9-Star))

#과제 - 구구단
for left in range(2,10):
    if (left % 2 == 0):
        for right1 in [1,3,5,7,9]:
            print(left,"*",right1,"=",left*right1)
    else:
        for right2 in [2,4,6,8]:
            print(left,"*",right2,"=",left*right2)

#과제 - 소수 찾기
num = int(input("소수를 구하려는 범위를 입력(2 이상) : "))
List = [2]
for Number in range(3,num+1):
    if (num == 2):
        break
    elif (num >= 3):
        for Qu in range(2,Number):
            if (Number % Qu == 0):
                break
            elif (Qu == Number-1):
                List.append(Number)
            else:
                continue
    else:
        print("2이상 입력")
        continue
print(List)

#과제 - 피보나치 수열
fibo_list = [1,1]
An = int(input("피보나치 수열의 항의 수를 입력 : "))
for Fi in range(2,An) :
    fibo_list.append(fibo_list[Fi-1] + fibo_list[Fi-2])
print(fibo_list)
