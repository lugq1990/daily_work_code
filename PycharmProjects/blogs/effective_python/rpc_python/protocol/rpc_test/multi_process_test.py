import os
from multiprocessing import Pool

def f(x):
    print("Parent ID:", os.getppid())
    print("Current PID:", os.getpid())
    return x**2

if __name__ == "__main__":
    with Pool(5) as p:
        p.map(f, [1, 2, 3, 5])