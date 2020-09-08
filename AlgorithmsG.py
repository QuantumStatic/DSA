from math import sqrt, ceil, floor, log
from myFunctionsG import execute_this, nearMatching
from random import randint as rint
from statistics import mean
from functools import lru_cache


def polynomial_eval(polynomial, value):
        for _ in range(power):
            pow_calc *= value
        evaluation += coefficient * pow_calc
    return (evaluation)


def horner_itr(polynomial, value):
    # bottoms-up implementation
    ''' horner algorithms aims to remove the dependency on the pow function or calculation the traditional 
        way by breaking it into self similar pieces (in a way breaking into a puzzle made up of itself) '''

    '''y = a(k) + y*x is the horner algorithm. where k ranges from n to 0. in the first iteration y just equals to the 
    innermost constant and nothing else. That is an implication of the self refrential equation not a planned occurence  '''

    summation = int()
    for index in reversed(polynomial):
        summation = index + value * summation

    return (summation)


def horner_recur(polynomial, value):
    # top-down implementation

    if len(polynomial) == 1:
        return polynomial.pop()

    coeff = polynomial.pop(0)
    return (coeff + value * horner_recur(polynomial, value))


def horner_alt(polynomial, value):

    evaluation = int()
    for coefficient, power in enumerate(polynomial):
        evaluation += coefficient * pow(value, power)
    return (evaluation)


def fibPhi(n):
    phi = (1 + sqrt(5)) / 2
    return floor(pow(phi, n)/sqrt(5)+0.5)

@lru_cache(None)
def fibCache(n):
    if n < 2:
        return n
    return fibCache(n - 1) + fibCache(n - 2)

def find_max_sub_array_recur(arrey):

    # this is found by running brute force against the recursive implementation. Depends on hardware specifications.
    recur_better_than_brute_point = 8

    def find_max_sub_array_brute(arrey):
        maxSum, start, end = float("-Inf"), int(), int()
        for x in range(len(arrey)):
            currSum = int()
            for y in range(x, len(arrey)):
                currSum += arrey[y]
                if currSum > maxSum:
                    start, end, maxSum = x, y, currSum
        return (start, end, maxSum)

    def find_Max_Mid(arr, begin, mid, end):
        leftSum, currSum, maxLeft = float("-Inf"), int(), int()
        for x in range(mid, begin - 1, -1):
            currSum += arr[x]
            if currSum > leftSum:
                leftSum = currSum
                maxLeft = x
        rightSum, currSum, maxRight = float("-Inf"), int(), int()
        for x in range(mid+1, end + 1):
            currSum += arr[x]
            if currSum > rightSum:
                rightSum = currSum
                maxright = x

        return (maxLeft, maxRight, leftSum + rightSum)

    def find_Max(arr, begining, end):
        if begining is end:
            return (begining, end, arr[end])
        elif begining - end > recur_better_than_brute_point:
            mid = (begining + end) // 2
            leftBegin, leftEnd, leftSum = find_Max(arr, begining, mid)
            rightBegin, rightEnd, rightSum = find_Max(arr, mid + 1, end)
            midBegin, midEnd, midSum = find_Max_Mid(arr, begining, mid, end)
            if leftSum >= rightSum and leftSum >= midSum:
                return (leftBegin, leftEnd, leftSum)
            elif leftSum < rightSum and rightSum >= midSum:
                return (rightBegin, rightEnd, rightSum)
            else:
                return (midBegin, midEnd, midSum)
        else:
            return find_max_sub_array_brute(arrey)

    return find_Max(arrey, 0, len(arrey) - 1)


def find_max_sub_array_linear(arrey):
    ''' for a better understanding of this algorthim, plot y = f(x), where f(x) is defined as sum of all 
    elements of a list from x =0 to x =n. This algorthim finds the area under the graph when f(n) >= 0 '''

    maxSum, currSum, begin, end = float("-Inf"), float("-Inf"), int(), int()
    for x in range(len(arrey)):
        currEnd = x
        if currSum > 0:
            currSum += arrey[x]
        else:
            currBegin, currSum = x, arrey[x]
        if currSum > maxSum:
            maxSum, begin, end = currSum, currBegin, currEnd

    return (begin, end, maxSum)


def extractMaxMin(arg):
    # Following code does 3 comparisions for every 2 elements, rather than intutive way of 2 comparisions per element
    n, Max, Min = len(arg), int(), int()
    if n % 2:
        Max = Min = arg[0]
        for x in range(1, n-1, 2):
            big, SMALL, n1, n2 = int(), int(), arg[x], arg[x+1]
            if n1 > n2:
                big, SMALL = n1, n2
            else:
                big, SMALL = n2, n1
            if big > Max:
                Max = big
            if SMALL < Min:
                Min = SMALL 
    else:
        Max = Min = float("-inf")
        for x in range(0, n-1, 2):
            big, SMALL, n1, n2 = int(), int(), arg[x], arg[x+1]
            if n1 > n2:
                big, SMALL = n1, n2
            else:
                big, SMALL = n2, n1
            if big > Max:
                Max = big
            if SMALL < Min:
                Min = SMALL


def IorderSelection(List, order):

    def Partition(List, lowEnd, highEnd):
        leftIndex, rightIndex, tempList = lowEnd, highEnd - 1, List[lowEnd:highEnd]
        pivot = List[nearMatching(tuple(tempList), mean(tempList))+lowEnd]
        while True:
            while rightIndex >= lowEnd and List[rightIndex] >= pivot:
                rightIndex -= 1
            while leftIndex < highEnd and List[leftIndex] <= pivot:
                leftIndex += 1
            if leftIndex < rightIndex:
                List[leftIndex], List[rightIndex] = List[rightIndex], List[leftIndex]
            else:
                return rightIndex

    def Select(List, lowEnd, highEnd):
        if lowEnd < highEnd:
            positionedElement = Partition(List, lowEnd, highEnd)
            if positionedElement == order:
                return List[positionedElement]
            elif order < positionedElement:
                return Select(List, lowEnd, positionedElement)
            else:
                return Select(List, positionedElement+1, highEnd)
        return
    
    return Select(List, 0, len(List))
