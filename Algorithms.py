from math import sqrt, gcd, floor, ceil
from myFunctions import execute_this, nearMatching, compare_these
from random import randint as rint
from statistics import mean
from functools import lru_cache




def polynomial_eval(polynomial, value):
    evaluation = int()
    for coefficient, power in enumerate(polynomial):
        pow_calc = int(1)
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

    def power(a,b):
        final =0
        while (b):
            if b & 1:
                final *= a
            a *= a
            b //= 2
        return final

    evaluation = int()
    for coefficient, powe in enumerate(polynomial):
        evaluation += coefficient * power(value, powe)
    return (evaluation)


def fibPhi(n):
    phi = (1 + sqrt(5)) / 2
    return floor(pow(phi, n)/sqrt(5)+0.5)

@lru_cache(None)
def fibCache(n):
    if n < 2:
        return n
    return fibCache(n - 1) + fibCache(n - 2)


def dynamicFib(n):
    register = dict()
    register[0], register[1] = 0, 1
    
    def fib(n):
        try:
            return register[n]
        except:
            register[n] = fib(n-1) + fib (n-2)
            return register[n]
    
    return fib(n)


def fibMatrixExp(n):
    # any fibonacci number at nth postition can be written as as (n-1)th power of the matrix [[1,1][1,0]] mulitplied by [fibonaaci(1), fibonacci(0)]
    result = tuple([[1,0],[0,1]])
    OG = tuple([[1,1],[1,0]])
    n -= 1

    def matrixMultiply(a,b):
        fin = tuple([[0,0],[0,0]])
        fin[0][0] = (a[0][0]*b[0][0] + a[0][1]*b[1][0]) #% 10
        fin[0][1] = (a[0][0]*b[0][1] + a[0][1]*b[1][1]) #% 10
        fin[1][0] = (a[1][0]*b[0][0] + a[1][1]*b[1][0]) #% 10
        fin[1][1] = (a[1][0]*b[0][1] + a[1][1]*b[1][1]) #% 10
        return fin
    
    while n:
        if n & 1: 
            result = matrixMultiply(OG,result) 
        OG = matrixMultiply(OG,OG) 
        n //= 2
    
    return matrixMultiply(result,tuple([[1,0],[0,1]]))[0][0]


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
                maxRight = x

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


def longestCommonSubsequence(seq1:str, seq2:str, only_length = False):
    seq1_len, seq2_len = len(seq1), len(seq2)
    matching_sequence_table = list() 
    
    if not only_length:
        for _ in range(seq1_len + 1):
            matching_sequence_table.append([0] * (seq2_len + 1))
        
        for x in range(1, seq1_len + 1):
            for y in range(1, seq2_len + 1):
                if seq1[x-1] == seq2[y-1]:
                    matching_sequence_table[x][y] = 1 + matching_sequence_table[x-1][y-1]
                else:
                    matching_sequence_table[x][y] = max(matching_sequence_table[x-1][y], matching_sequence_table[x][y-1])
        
        LCS_len = matching_sequence_table[seq1_len][seq2_len]
        longest_common_subsequence = [0] * LCS_len
        x, y = seq1_len, seq2_len
        while matching_sequence_table[x][y] != 0:
            if matching_sequence_table[x][y - 1] != matching_sequence_table[x][y]:
                longest_common_subsequence[LCS_len - 1] = seq2[y - 1]
                x -= 1
                LCS_len -= 1
            y -= 1

        return ''.join(longest_common_subsequence)
    
    else:
        for _ in range(2):
            matching_sequence_table.append([0] * (seq2_len + 1))

        for _ in range(seq1_len):
            for y in range(1, seq2_len + 1):
                if seq1[0] == seq2[y-1]:
                    matching_sequence_table[1][y] = 1 + matching_sequence_table[0][y-1]
                else:
                    matching_sequence_table[1][y] = max(matching_sequence_table[0][y], matching_sequence_table[1][y-1])
            seq1 = seq1[1:]
            matching_sequence_table[0] = matching_sequence_table[1]

        return matching_sequence_table[1][seq2_len]


def binaryExponentiation(a: int, b: int, modulo: int = None):
    
    result = 1
    if modulo is None:
        modulo = a + 1
    
    while b:
        if b & 1:
            result *= a
            result %= modulo
        a *= a
        a %= modulo
        b //= 2
    
    return result


def jugProblem(jug1:int, jug2:int, toMeasure:int, justSteps = False):
    smallerJugSize, biggerJugSize, jugToFill =  jug1, jug2, toMeasure
    
    if jugToFill % gcd(smallerJugSize, biggerJugSize) is not 0:
        print("No Solution")
        return
    
    if smallerJugSize > biggerJugSize:
        smallerJugSize, biggerJugSize = biggerJugSize, smallerJugSize

    steps, steps_taken = int(), list()
    smallJug = bigJug =  int()
    while bigJug != jugToFill and smallJug != jugToFill:
        if bigJug is 0:
            bigJug = biggerJugSize
        elif smallJug == smallerJugSize:
            smallJug = 0
        else:
            spaceInSmallerJug = smallerJugSize - smallJug
            smallJug, bigJug = smallJug + (spaceInSmallerJug if bigJug >= spaceInSmallerJug else bigJug), bigJug - (spaceInSmallerJug if bigJug >= spaceInSmallerJug else bigJug)
        steps_taken.append((smallJug, bigJug))
        steps += 1

    if justSteps:
        return steps
    else:
        return steps_taken


def constructionProblem(toConstruct:int, pieces:tuple, total_ways:bool = False, minimum_pieces:bool = True, sequence:bool = False) -> int:
    constructionTable = list()
    nOfPieces = len(pieces)
    
    if total_ways:
        for _ in range(2):
            constructionTable.append(list())
        
        if pieces[0] is 0:
            constructionTable[0].append(1)
            constructionTable[0].extend([0 for _ in range(toConstruct)])
        else:
            for x in range(toConstruct+1):
                constructionTable[0].append(1 if x % pieces[0] == 0 else 0)
        constructionTable[1].extend([0 for _ in range(toConstruct+1)])
        
        pieces = pieces[1:]
        
        for _ in range(nOfPieces-1):
            for y in range(toConstruct+1):
                ways = constructionTable[0][y]
                if pieces[0] <= y:
                    ways += constructionTable[1][y - pieces[0]]
                constructionTable[1][y] = ways
            pieces = pieces[1:]
            constructionTable[0] = constructionTable[1]

        return constructionTable[1][toConstruct]
    
    else:
        if sequence:
            for _ in range(nOfPieces):
                constructionTable.append(list())
            
            if pieces[0] is 0:
                constructionTable[0].append(1)
                constructionTable[0].extend([0 for _ in range(toConstruct)])
            else:
                for x in range(toConstruct+1):
                    constructionTable[0].append(x // pieces[0] if x % pieces[0] == 0 else 0)

            for x in range(1, nOfPieces):
                for y in range(toConstruct+1):
                    ways = constructionTable[x-1][y]
                    if pieces[x] <= y:
                        ways = min(1 + constructionTable[x][y - pieces[x]], ways)
                    constructionTable[x].append(ways)
            
            x = nOfPieces - 1 
            y = toConstruct
            sequence = list()
            while x is not 0:
                if constructionTable[x][y] != constructionTable[x-1][y]:
                    sequence.append(pieces[x])
                    y -= pieces[x]
                else:
                    x -= 1

            return tuple(sequence)

        else:
            for _ in range(2):
                constructionTable.append(list())
        
            if pieces[0] is 0:
                constructionTable[0].append(1)
                constructionTable[0].extend([0 for _ in range(toConstruct)])
            else:
                for x in range(toConstruct+1):
                    constructionTable[0].append(1 if x % pieces[0] == 0 else 0)
            constructionTable[1].extend([0 for _ in range(toConstruct+1)])
            
            pieces = pieces[1:]
            
            for _ in range(nOfPieces-1):
                for y in range(toConstruct+1):
                    ways = constructionTable[0][y]
                    if pieces[0] <= y:
                        ways = min(constructionTable[1][y - pieces[0]], ways)
                    constructionTable[1][y] = ways
                pieces = pieces[1:]
                constructionTable[0] = constructionTable[1]

            return constructionTable[1][toConstruct]

def primeGenerator(limit):
    
    isprime, smallestPrimeFactor, primes = [False] * 2 + [True] * (limit - 2), [int()] * limit, list()
    primes_found = int()
    for i in range(2, limit): 
        if isprime[i]: 
            primes.append(i) 
            smallestPrimeFactor[i] = i 
            primes_found += 1
        j = int()
        while j < primes_found and i * primes[j] < limit and primes[j] <= smallestPrimeFactor[i]:
            isprime[i * primes[j]] = False
            smallestPrimeFactor[i * primes[j]] = primes[j]
            j += 1
    return primes

def prime_checker(suspected_prime):
    # Checking primes since '99. supports lists and individual numbers as well
    if isinstance(suspected_prime, list):
        dummy = list()
        for prime_candidate in suspected_prime:
            dummy.append(prime_checker(prime_candidate))
        return dummy
    else:
        suspected_prime = abs(suspected_prime)
        if suspected_prime == 1 or suspected_prime == 2 or suspected_prime == 3:
            return (False if suspected_prime == 1 else True)
        if suspected_prime % 2 == 0 or suspected_prime % 3 == 0:
            return False
        end_point, prime_factor = ceil(suspected_prime**0.5), 5
        while True:
            if end_point < prime_factor:
                return True
            if suspected_prime % prime_factor == 0 or suspected_prime & (prime_factor + 2) == 0:
                return False
            prime_factor += 6

def sieveErato(limit):
    # Sieve of Eratothenes. Looks up prime numbers upto almost 8 million in a second.
    if limit <= 90_000:
        primes, index, endPoint, result = [False, True] * (limit//2+1), 3, ceil(limit**0.5) + 1, [2]
        while index <= endPoint: #sqrt of limit is the endpoint
            for compositeNum in range(index ** 2, limit + 1, index * 2): 
                primes[compositeNum] = False
            index += 2
            while not primes[index]:
                index += 2
        for x in range(3, len(primes), 2):
            if primes[x]:
                result.append(x)
        return (result)
    else:
        primes = list()
        for x in range(limit):
            if prime_checker(x):
                primes.append(x)
        return primes

def prime_factoriser(n):
    # I am a rookie hence this implementation. upgrade due, feel free to suggest improvements

    if prime_checker(n):
        return ([n])
    prime_factor, list_of_factors = 2, list()
    while n % prime_factor == 0:
        n //= prime_factor
        list_of_factors.append(prime_factor)
    if n == 1:
        return (list_of_factors)
    end_point, prime_factor = ceil(sqrt(n)), 3
    while prime_factor < end_point+1:
        if n % prime_factor == 0:
            n //= prime_factor
            list_of_factors.append(prime_factor)
            if n == 1:
                return(list_of_factors)
        else:
            prime_factor += 2
    list_of_factors.append(n)
    return(list_of_factors)


@execute_this
def main():
    print(constructionProblem(200, [1,2,5,10,20,25,50,100,150]))


