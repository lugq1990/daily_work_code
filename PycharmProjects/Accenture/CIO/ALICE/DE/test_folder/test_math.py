# -*- coding:utf-8 -*-
import math
def quaratic(a, b, c):
    # if b**2 - 4 * a * c < 0, shouldn't run bellow code!
    if (b**2 - 4 * a * c) < 0:
        print("You couldn't make the b*b - 4*a*c bellow 0!")
        exit()
    x1 = 1/(2 * a) * (-b + math.sqrt((b**2 - 4 * a * c)))
    x2 = 1 / (2 * a) * (-b - math.sqrt((b ** 2 - 4 * a * c)))
    return x1, x2

