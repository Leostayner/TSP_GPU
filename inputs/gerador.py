import math
import random

if __name__ == "__main__":
    N = int(input())
    print(N)
    sz = random.randint(1000, 10000)

    for i in range(N):
        x = random.random() * sz
        y = random.random() * sz
        print(x, y)