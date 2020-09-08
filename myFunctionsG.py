# working directory is the location where the file is saved
from time import process_time_ns
from math import ceil, sqrt, floor, gcd
from itertools import permutations
from collections import Counter
import cmath
import random


def execution_time(total_time):
    # I get an OCD if it says 1 seconds and not 1 second hence this.

    ms, total_time = int(), total_time / 10**6
    mins = int(total_time // 60000)
    secs = int(total_time // 1000 - mins * 60)
    prefix, Min, Minz, Sec, Secx, Msec, Msecs, Mz, Mzs, suffix = "Execution Time: ", " Minute ", " Minutes ", " Second ", " Seconds ", " millisecond ", " milliseconds ", " microsecond ", " microseconds ", '\n'

    if total_time < 100:
        if total_time < 1:
            ms = round(total_time, 3)
        elif total_time < 10:
            ms = round(total_time, 2)
        else:
            ms = round(total_time, 1)
        if ms.is_integer():
            ms = int(ms)
    else:
        ms = int(round(total_time - (secs * 1000 + mins * 60000), 0))

    finalPrint = prefix
    if mins > 0:
        finalPrint += f"{mins}"
        if mins == 1:
            finalPrint += Min
        else:
            finalPrint += Minz
    if secs > 0:
        finalPrint += f"{secs}"
        if secs == 1:
            finalPrint += Sec
        else:
            finalPrint += Secx
    if ms > 0:
        if total_time < 1:
            ms = round(ms*100, 2)
            try:
                if ms.is_integer():
                    ms = int(ms)
            except:
                pass
            finalPrint += f"{ms}"
            if ms == 1:
                finalPrint += Mz
            else:
                finalPrint += Mzs
        else:
            finalPrint += f"{ms}"
            if ms == 1:
                finalPrint += Msec
            else:
                finalPrint += Msecs
    finalPrint += suffix
    print(finalPrint)


def sieveErato(limit):
    # Sieve of Eratothenes. Looks up prime numbers upto almost 8 million in a second.

    primes, index, endPoint, result = [False, True] * (limit//2+1), 3, ceil(limit**0.5) + 1, [2]
    while index <= endPoint:
        for compositeNum in range(index ** 2, limit + 1, index * 2):
            primes[compositeNum] = False
        index += 2
        while not primes[index]:
            index += 2
    for x in range(3, len(primes), 2):
        if primes[x]:
            result.append(x)
    return (result)


def prime_checker(suspected_prime):
    # Checking primes since '99. supports lists and individual numbers as well
    if isinstance(suspected_prime, list):
        dummy = list()
        for prime_candidate in suspected_prime:
            dummy.append(prime_checker(prime_candidate))
        return (dummy)
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
            if suspected_prime % prime_factor == 0 or suspected_prime % (prime_factor+2) == 0:
                return False
            prime_factor += 6


def remove_duplicates_from(input_with_duplicates):
    # it's bad but it's mine, open to improvements

    if isinstance(input_with_duplicates, list):
        return(list(dict.fromkeys(input_with_duplicates)))
    else:
        return(combine_list(remove_duplicates_from(split_stuff(input_with_duplicates))))


def max_consecutive_elements_counter(array, element_to_be_checked):
    # old code, up for maintenance and improvement. 
    consec_counter, final_answer = 0, 0
    for x in array:
        if x == element_to_be_checked:
            consec_counter += 1
        else:
            final_answer, consec_counter = (consec_counter if consec_counter > final_answer else final_answer), int()
    return (final_answer if final_answer > consec_counter else consec_counter)


def split_stuff(stuff, return_type=None):
    # too lazy to write this everytime.
    dummyList = list()
    if isinstance(stuff, int):
        while stuff:
            dummyList.insert(0, stuff % 10)
            stuff //= 10
    if isinstance(stuff, str):
        dummyList.extend([char for char in stuff])
    if return_type is not None:
        return (map(return_type, dummyList))
    else:
        return (iter(dummyList))


def combine_list(list_to_combine):
    # sometimes you just want elemnts of a list to be combined. It's not used often but does save lives when it matters

    if isinstance(list_to_combine[0], str):
        string = str()
        return(string.join(list_to_combine))
    if isinstance(list_to_combine[0], int):
        num = int()
        for x in list_to_combine:
            num = num*10 + x
        return(num)
    if isinstance(list_to_combine[0], list) or isinstance(list_to_combine[0], tuple):
        dummy = list()
        for element in list_to_combine:
            dummy.append(combine_list(element))
        return (dummy)


def list_substractor(main_list, list_to_substract):
    # I don't remember half this file. I proabably was too lazy and just wrote it thinking that I might it need it later

    list_to_substract = set(list_to_substract)
    return(list(filter(lambda x: x not in list_to_substract, main_list)))


def find_pythogorean_triplets_till(num):
    # bad code. improvemnt known. will be updated soon to run in linear time

    list_of_triplets = list()
    for m in range(num+1):
        for n in range(m):
            a, b, c = m**2 - n**2, 2*m*n, m**2 + n**2
            list_of_triplets.extend([a, b, c])
    yield(list_of_triplets)


def check_list_containment(main_list, list_to_check, method1=None, method2=None):
    # set method removes duplicates before checking can be a pain someitmes, hence my code. looking for imrpovements
    if method1:
        return(set(list_to_check).issubset(set(main_list)))
    elif method2:
        listUnderScrutiny = iter(list_to_check)
        for x in listUnderScrutiny:
            if x in main_list:
                main_list.remove(x)
            else:
                return False
        return True
    else:
        raise Exception("No method found to be specified")


def execute_this(code_to_execute):
    # I was tired of importing time and then making a start variable, calling the execution time function, I just wrote this
    print(f"Running {code_to_execute.__name__}\n")
    start = process_time_ns()
    code_to_execute()
    execution_time(process_time_ns()-start)


def LCM(a, b):
    # Until math.LCM comes in 3.9, this is my code
    return (a*b//gcd(a, b))


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


def sieveEratoAlt(limit):
    # Lazy Evaluation of the SieveErato method. sometimes I can use the generator when simeply iterating over it so why not.

    primes, index, endPoint = [False, True] * \
        (limit//2), 3, ceil(limit**0.5) + 1,
    while index <= endPoint:
        for compositeNum in range(index ** 2, limit + 1, index * 2):
            primes[compositeNum] = False
        index += 2
        while not primes[index]:
            index += 2
    for x in range(3, len(primes), 2):
        if primes[x]:
            yield (x)


def nearMatching(List, target):
    # this function returns the index of the element closest to target, in value
    differneces = tuple(map(lambda x: abs(x - target), List))
    return differneces.index(min(differneces))
