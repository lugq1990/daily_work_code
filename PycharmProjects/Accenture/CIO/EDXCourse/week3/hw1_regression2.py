import numpy as np
import sys

lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

## Solution for Part 1
def part1():
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    x, y = np.array(X_train), np.array(y_train).reshape(-1, 1)

    tmp = np.dot(x.T, x)
    i_m = np.identity(len(tmp))
    out = np.dot(np.dot(np.matrix(tmp + lambda_input * i_m).I, x.T), y)

    return np.array(out.reshape(-1))


wRR = part1()  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file

## Solution for Part 2
def part2():
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    # first just get 10 samples from test datasets
    pass


active = part2()  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file