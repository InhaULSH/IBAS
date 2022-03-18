class Student:
    name = str()
    std_no = int()
    grade_dict = dict()
    def __init__(self, Name, Number, Grade):
        self.name = Name
        self.std_no = Number
        self.grade_dict = Grade
    def std_sum(self):
        SumofDict = 0
        SumofDict += self.grade_dict['국어']
        SumofDict += self.grade_dict['수학']
        SumofDict += self.grade_dict['과학']
        SumofDict += self.grade_dict['사회']
        SumofDict += self.grade_dict['영어']
        return SumofDict
    def std_avg(self):
        Sum = float(self.std_sum())
        Avg = round(Sum / 5, 1)
        return Avg

class SMS:
    inst_list = []
    def __init__(self, Inst):
        self.inst_list = Inst
    def all_sum(self):
        SumList = 0
        for i in range(len(self.inst_list)):
            SumList += self.inst_list[i].std_avg()
        return SumList
    def all_avg(self):
        SumList = float(SMS.all_sum(self))
        Lenlist = float(len(self.inst_list))
        Avg = round(SumList / Lenlist, 1)
        return Avg

Command = 0
ListofInstance = []
while (Command != 4) :
    Command = int(input("숫자로 명령어를 입력하세오.\n1.조회 2.추가 3.삭제 4.프로그램 종료\n"))
    if (Command == 2) :
        StudentName = str(input("학생의 이름을 입력하세요 : "))
        StudentNumber = int(input("학생의 학번을 입력하세요 : "))
        for X in range(len(ListofInstance)) :
            if (ListofInstance[X].std_no == StudentNumber) :
                OverlapCommand = input("이미 같은 학번의 학생이 존재합니다. 새로 입력하는 정보로 덮어쓰시겠습니까? (Y/N) : ")
                if (OverlapCommand == 'Y') :
                    for X in range(len(ListofInstance)):
                        if (ListofInstance[X].std_no == StudentNumber):
                            ListofInstance.pop(X)
                        else:
                            continue
                elif (OverlapCommand == 'N') :
                    print("덮어쓰지 않고 기존 학생과 다른 학생의 정보로 간주합니다.")
                else :
                    print("잘못된 입력입니다. 덮어쓰지 않는 것으로 간주합니다.")
        StudentKorean = int(input("국어 성적을 입력하세요 : "))
        StudentMath = int(input("수학 성적을 입력하세요 : "))
        StudentSocial = int(input("사회 성적을 입력하세요 : "))
        StudentScience = int(input("과학 성적을 입력하세요 : "))
        StudentEnglish = int(input("영어 성적을 입력하세요 : "))
        StudentInfo = Student(StudentName, StudentNumber, {'국어': StudentKorean, '수학' : StudentMath, '사회' : StudentSocial, '과학' : StudentScience, '영어' : StudentEnglish})
        ListofInstance.append(StudentInfo)

    elif (Command == 1) :
        if (len(ListofInstance) == 0) :
            print("아직 학생 정보가 등록되지 않았습니다")
            continue
        SMSInfo = SMS(ListofInstance)
        for X in range(len(ListofInstance)) :
            print("학생의 이름 : ", ListofInstance[X].name, " | 국어 : ", ListofInstance[X].grade_dict['국어'], " | 수학 : ", ListofInstance[X].grade_dict['수학'], " | 사회 : ", ListofInstance[X].grade_dict['사회'], " | 과학 : ", ListofInstance[X].grade_dict['과학'], " | 영어 : ", ListofInstance[X].grade_dict['영어'], " | 총점 : ", ListofInstance[X].std_sum(), " | 평균 : ", ListofInstance[X].std_avg())
        print("전체 평균 : ", round(SMSInfo.all_avg(), 1))
    elif (Command == 3) :
        StudentDelete = int(input("리스트에서 제거할 학생의 학번을 입력하세요 : "))
        for X in range(len(ListofInstance)):
            if (int(ListofInstance[X].std_no) == StudentDelete) : #여기서 ListofInstance[X].std_no에 int를 씌우지 않으면 오류가 나던데 왜 그런 건가요? 궁금합니다
                ListofInstance.pop(X)
    elif (Command == 4) :
        print("프로그램을 종료합니다")
    else :
        print("명령을 잘못 입력했습니다. 올바른 명령어를 입력하세요")