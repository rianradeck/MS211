import numpy as np
from tabulate import tabulate as tbl
import matplotlib.pyplot as plt
from numpy import linalg as LA

def print_table(T, headers = []):
    print(tbl(T, headers, tablefmt = "fancy_grid", floatfmt = ".4f"))

def get_A(x, g):
    return np.array([g(x_) for x_ in x])

def least_square_method(x, y, g):
    A = get_A(x, g)
    print("A")
    print_table(A)
    At = np.transpose(A)
    # print_table(At)

    B = At @ A
    b = At @ y

    print_table(B)
    print_table(b)

    alpha = LA.solve(B, b)
    print_table(alpha)

    X = x.copy()
    Y = y.copy()

    def f(x):
        ans = 0
        for i in range(len(alpha)):
            ans += g(x)[i] * alpha[i]
        # return x * alpha[0] + alpha[1]
        return ans

    x_l = np.linspace(x.min(), x.max(), 100) # line aproximation (the higher the last number better the graph resolution will be)
    y_l = np.array([f(x_) for x_ in x_l])

    print(y.shape)
    residue = np.array([(f(X[i]) - y[i][0]) ** 2 for i in range(len(X))]).sum()
    print("residue", residue)

    # print(x, "\n", y)

    plt.scatter(X, Y, color = "red") # original info
    plt.plot(x_l, y_l, color = "blue") # aproximation found
    plt.show()

    return alpha

# DADOS DA LETRA A (12-month)
xA = np.arange(2011, 2021)
yA = np.transpose(np.array([[9.51, 9.6, 9.63, 9.71, 9.92, 9.93, 9.79, 9.78, 10.07, 10.13]]))

# DADOS DA LETRA E (10-year)
xE = np.arange(2006, 2016)
yE = np.transpose(np.array([[9.55, 9.55, 9.56, 9.58, 9.59, 9.64, 9.65, 9.69, 9.72, 9.77]]))

alpha = least_square_method(xE, yE, lambda K: np.array([K, 1]))

def f(x):
    return x * alpha[0] + alpha[1]

true_values = np.array([9.93, 9.79, 9.78, 10.07, 10.13])
estimative = np.array([f(k) for k in range(2016, 2021)]).reshape(len(true_values))
diff = true_values - estimative

print(true_values)
print(estimative)
print(diff.mean(), diff.std())

x = np.arange(1, 35, 3)
y = np.transpose(np.array([[23.6, 21.7, 18.3, 20.8, 25.6, 22.6, 17.1, 22.8, 23.8, 21, 18.8, 22.3]]))

alpha = least_square_method(x, y, lambda K: np.array([1, np.cos(2 * np.pi/12 * K), np.sin(2 * np.pi / 12 * K)]))
print(alpha)