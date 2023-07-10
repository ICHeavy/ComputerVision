s = "abccccdd"

singles = dict()
for i in s:
#   if it is in the dict that means it is a duplicate
#   so we can remove it, knowing that if it comes up again it can be put in the middle
    if i in singles:
        print(i, 'already in dict')
        del singles[i]
    else:
# beginning here add char to list, if it is seen again it will be removed from the list
        singles[i] = 1
        print(singles)

# subbing singles from total results in number of couples aka len of palindrome
# if singles is empty theres an even amt of numbers
if len(singles) == 0:
    print( len(s) - len(singles))
else:
    print( len(s) - len(singles) + 1)