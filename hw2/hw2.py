import sys
import math
import os

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=dict.fromkeys({chr(i): 0 for i in range(ord('A'), ord('Z') + 1)},0)

    if not os.path.isfile(filename):
        print(f"Error: '{filename}' does not exist.")
        sys.exit(1)

    with open (filename,encoding='utf-8') as f:
        # TODO: add your code here
        text = f.read()

    for char in text:
        if char.isalpha():
            char = char.upper() #case-folding
            X[char] = X.get(char, 0)+1 # Set the value to 1 if key doesn't exist, otherwise increment
            # output doesn't contain letter with 0 count, X should be initialized

    print("Q1")
    # printprint(X.items()) not required format
    for letter in sorted(X.keys()):
        print(f"{letter} {X[letter]}")

    return X

# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!


def compute_log(X, e, s):
    X1 = X['A']
    e1 = e[0]
    s1 = s[0]

    X1_log_e1 = X1 * math.log(e1) if e1 > 0 else 0
    X1_log_s1 = X1 * math.log(s1) if s1 > 0 else 0

    print("Q2")
    print(f"{X1_log_e1:.4f}")
    print(f"{X1_log_s1:.4f}")


def compute_F(X, e, s, prior_e, prior_s):
    # F(y) = log(P(Y=y)) + sum(Xi * log(pi))

    F_e = math.log(prior_e)
    F_s = math.log(prior_s)

    for i, letter in enumerate(chr(j) for j in range(ord('A'), ord('Z') + 1)):
        Xi = X[letter]
        ei = e[i]
        si = s[i]

        if ei > 0:
            F_e += Xi * math.log(ei)
        if si > 0:
            F_s += Xi * math.log(si)

    print("Q3")
    print(f"{F_e:.4f}")
    print(f"{F_s:.4f}")

    return F_e, F_s


def compute_P(F_e, F_s):
    if F_s - F_e >= 100:
        P_e_X = 0
    elif F_s - F_e <= -100:
        P_e_X = 1
    else:
        P_e_X = 1 / (1 + math.exp(F_s - F_e))

    print("Q4")
    print(f"{P_e_X:.4f}")


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Error! [letter file] [english prior] [spanish prior] Needed!")
        sys.exit(1)

    letterfile = sys.argv[1]
    prior_e = float(sys.argv[2])
    prior_s = float(sys.argv[3])

    X = shred(letterfile)

    e, s = get_parameter_vectors()

    compute_log(X, e, s)

    F_e, F_s = compute_F(X, e, s, prior_e, prior_s)

    compute_P(F_e, F_s)