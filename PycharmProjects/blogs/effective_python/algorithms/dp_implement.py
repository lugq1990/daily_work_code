"""Try to implement a dynamic programming for coins [1,2,5] to get 3, how many coins at least"""

import numpy as np

def coin_change(amount, coins=[1, 2, 5]):
    mem = {}
    def dp(n):
        if n == 0:
            return 0
        if n < 0:
            return -1
        res = float("inf")

        for coin in coins:
            print("Coin: {}".format(coin))
            sub_problem = dp(n - coin)
            if sub_problem == -1:
                continue
            res = min(res, 1 + sub_problem)

        mem[n] = res
        if res == float('inf'):
            return -1
        return mem[n]

    return dp(amount)


def coin_change_dq(amount, coin_list=[1, 2, 5]):
    res = np.ones((amount + 1), dtype=int) * (amount + 1)
    
    for i in range(len(res)):
        for coin in coin_list:
            if i - coin < 0:
                continue
            res[i] = min(res[i], 1 + res[i - coin])

    if res[amount] == amount + 1:
        return -1
    return res[amount]

if __name__ == '__main__':
    print("best value:", coin_change(5))
    print("best value:", coin_change_dq(5))