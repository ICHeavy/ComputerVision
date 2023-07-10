
def countEven(self, num: int) -> int:
    count = 0
    temp = 0
    while num > 1:
        s = str(num)
        for i in s:
            temp += int(i)
            if temp % 2 == 0:
                count +=1
            else: 
                pass
    num -= 1
    return count


num = 5
print(countEven(num))