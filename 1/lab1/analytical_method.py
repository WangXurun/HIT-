import numpy as np
import util.util as util
import matplotlib.pyplot as plot
import math


def f(x):
    return np.sin(2 * np.pi * x)


# 样本点的数量
n = 7
# 方程的次数
order_ = 6

order = order_ + 1
lambd = 0

# 准备数据
x = np.linspace(0, 1, n)
X = np.asarray([np.power(x, i) for i in range(order)])
X = X.T
noise = np.random.normal(0, 0.5, (1, n))
Y = (f(x) + noise).T
# Y = np.asarray([[-0.5797487],
#                 [1.00653412],
#                 [0.5287666],
#                 [-0.46211614],
#                 [-0.39620244],
#                 [-0.86061157],
#                 [0.42344919]])

# 计算解析解
W = util.analytical(X, Y, lambd)
print(util.SSE(Y, np.dot(X, W)))

# 画图
X_ = np.linspace(0, 1, 100)
plot.title("analytical method, lambda= {},m = {}".format(lambd, order_))
plot.xlabel("X")
plot.ylabel("Y")
plot.scatter(x, Y.T, label="sample")
ans = [np.sum(list(W[i] * np.power(z, i) for i in range(order))) for z in X_]
plot.plot(X_, ans, label="output", color="blue")
plot.plot(X_, [np.sin(2 * np.pi * z) for z in X_], color="red", label="sin(2*pi*x)")
plot.legend()
plot.show()
