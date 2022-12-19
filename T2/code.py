import numpy as np
from numpy import linalg as LA
import tabulate

############### QUESTAO 1 ######################

def line_criteria(A):
	n = len(A)
	for i in range(n):
		alpha = 0
		for j in range(n):
			if i != j:
				print(i, abs(A[i][j]))
				alpha += abs(A[i][j])
		print(i, alpha, abs(A[i][i]))
		if alpha >= abs(A[i][i]):
			return False
	return True

def sassenfeld_criteria(A):
	n = len(A)
	gamma = np.zeros(n)
	for i in range(n):
		for j in range(i):
			print(i, A[i][j] * gamma[j])
			gamma[i] += A[i][j] * gamma[j]
		for j in range(i + 1, n):
			print(i, A[i][j])
			gamma[i] += A[i][j]
		print(i, gamma[i], A[i][i])
		if gamma[i] >= A[i][i]:
			return False
	return True

def jacobi(A, b, eps):
	n = len(A)

	D = np.zeros(n * n).reshape(n, n)
	C = np.zeros(n * n).reshape(n, n)
	for i in range(n):
		D[i][i] = A[i][i].copy()
		for j in range(n):
			if i != j:
				C[i][j] = A[i][j].copy()
	C = -LA.inv(D) @ C
	g = LA.inv(D) @ b

	print("C_j =\n", tabulate.tabulate(C, tablefmt="fancy_grid", floatfmt=".4f"), sep = '')
	print("Norm-inf (C_j) =", LA.norm(C, np.inf))

	k = 0
	x = [np.array([5, 2, 2, 4, 3, 2, 0, 11]).reshape(n, 1)]
	table = []
	table.append([0, x[0], "-", LA.norm((A @ x[0]) - b, np.inf)])
	while k == 0 or (LA.norm(x[k] - x[k - 1], np.inf) >= eps and LA.norm((A @ x[k]) - b, np.inf) >= eps):
		x.append(C @ x[k] + g)
		k += 1
		table.append([k, x[k], LA.norm(x[k] - x[k - 1], np.inf), LA.norm((A @ x[k]) - b, np.inf)])

	print(tabulate.tabulate(table, tablefmt="fancy_grid"))
	return x

def gs(A, b, eps, printtable = False):
	n = len(A)

	L = np.tril(A)
	U = np.triu(A, 1)
	
	C = -LA.inv(L) @ U
	g = LA.inv(L) @ b

	print("C_gs =\n", tabulate.tabulate(C, tablefmt="fancy_grid", floatfmt=".4f"), sep = '')
	print(LA.norm(C, np.inf))
	# print("g-gs\n", g)

	k = 0
	x = [np.array([5, 2, 2, 4, 3, 2, 0, 11]).reshape(n, 1)]
	table = []
	table.append([0, x[0], "-", LA.norm((A @ x[0]) - b, np.inf)])
	while k == 0 or (LA.norm(x[k] - x[k - 1], np.inf) >= eps and LA.norm((A @ x[k]) - b, np.inf) >= eps):
		x.append(C @ x[k] + g)
		k += 1
		table.append([k, x[k], LA.norm(x[k] - x[k - 1], np.inf), LA.norm((A @ x[k]) - b, np.inf)])
	
	if printtable:
		print(tabulate.tabulate(table, tablefmt="fancy_grid"))
	return x


A = np.array([[0.3, 0.05, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1], [0.1, 0.3, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0], [0.0, 0.2, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.1, 0.4, 0.0, 0.0, 0.0, 0.1], [0.0, 0.0, 0.1, 0.1, 0.55, 0.0, 0.0, 0.0], [0.0, 0.0, 0.1, 0.0, 0.0, 0.6, 0.2, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0], [0.2, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.5]])
d = np.array([[2.2], [0.5], [0.7], [1.0], [0.5], [0.4], [0.1], [4.0]])

M = np.identity(8) - A
print("################# LETRA D #####################")
x = gs(M, d, 1e-2)
print("################# LETRA E #####################")
x = jacobi(M, d, 1e-2)
print("################# LETRA G #####################")
x = gs(M, d, 1e-2, printtable = True)


################# QUESTAO 2 #####################

def G(x):
	return np.exp((x - 0.4) ** 2 / -0.002)

def T(x):
	return np.max(1 - (1/0.3)*abs(x - 0.6), 0)

def f(x):
	if x > 0.6:
		return -((1/0.3) * x) + 3 - np.exp(-((x - 0.4) ** 2/0.002))
	else:
		return (1/0.3) * x - 1 - np.exp(-((x - 0.4) ** 2/0.002))

def f_(x):
	if x > 0.6:
		return -3.33333 + (1000 * (-0.4 + x))/np.exp(500 * (-0.4 + x) ** 2)
	else:
		return +3.33333 + (1000 * (-0.4 + x))/np.exp(500 * (-0.4 + x) ** 2)

def newton(x):
	ans = []
	ans.append(x)
	for i in range(4):
		ans.append(ans[-1] - f(ans[-1]) / f_(ans[-1]))

	table = []
	table.append([0, ans[0], f(ans[0]), .0])
	for i in range(1, len(ans)):
		table.append([i, ans[i], f(ans[i]), np.abs(ans[i] - ans[i - 1])])
	print(tabulate.tabulate(table, tablefmt="fancy_grid", floatfmt=".8f"))

	return ans

# print(newton(0.4))
# print(newton(0.45))
# print(newton(0.5))
