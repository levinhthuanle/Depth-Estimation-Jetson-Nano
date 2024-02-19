def col(rowNumber, a): # Cột
    if a[0][rowNumber] == 0 or a[1][rowNumber] == 0:
        a[0][rowNumber] = a[1][rowNumber] = a[2][rowNumber] = 0

    return a 

def row(a): # Hàng
    for i in range(3):
        if a[2][i] == 0 and a[1][i] == 1:
            a[2][0] = a[2][1] = a[2][2] = 0
            return a
    return a

def printOutTheNumberOfRules(a):
    pass
    if a == [[1, 1, 1], [1, 1, 1], [1, 1, 1]]:
        return('1')
    
    if a == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]:
        return('2')

    if a == [[0, 1, 0], [0, 1, 0], [0, 1, 0]]:
        return('3')
    
    if a == [[0, 1, 1], [0, 1, 1], [0, 1, 1]]:
        return('4')

    if a == [[0, 0, 1], [0, 0, 1], [0, 0, 1]]:
        return('5')

    if a == [[1, 1, 0], [1, 1, 0], [1, 1, 0]]:
        return('6')

    if a == [[1, 0, 0], [1, 0, 0], [1, 0, 0]]:
        return('7')

    if a == [[1, 0, 1], [1, 0, 1], [1, 0, 1]]:
        return('8')

    if a == [[1, 1, 1], [1, 1, 1], [0, 0, 0]]:
        return('9')

    if a == [[0, 1, 1], [0, 1, 1], [0, 0, 0]]:
        return('10')

    if a == [[1, 1, 0], [1, 1, 0], [0, 0, 0]]:
        return('11')

    if a == [[0, 1, 0], [0, 1, 0], [0, 0, 0]]:
        return('12')
    
    return '1'
def specialCase(a):
    n = 2

    print("Before solving")
    for i in range(3):
        for j in range(3):
            print(a[i][j], end=' ')
        print()

    for i in range (3):
        for j in range(3):
            if a[i][j] >= 10:
                a[i][j] = 1
            else:
                a[i][j] = 0
    #
    for i in range(0, n + 1):
        a = col(i, a)
    a = row(a)

    print("After solving")
    for i in range(3):
        for j in range(3):
            print(a[i][j], end=' ')
        print()
    print(printOutTheNumberOfRules(a))
    return printOutTheNumberOfRules(a)