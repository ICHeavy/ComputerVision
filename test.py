import numpy as np
def fact(n: int) -> int:
    if n == 1:
        return n
    else:
        return n * fact(n-1)



def climbStairs( n: int) -> int:
    temp = 0
    for i in range(0,n-1):
        temp += fact(n)/(fact(i)*fact(n-i))
        
    
    return int(temp)

n = 3
print(climbStairs(n))