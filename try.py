num = int(input("Introduce un numero: "))
def numiter(num):
    print("Entering function")
    if num == 1:
        return num
    else:
        return numiter((num/2) +1)

numiter(num)
