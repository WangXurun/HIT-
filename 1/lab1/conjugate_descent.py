import numpy as np
import util.util as util
import matplotlib.pyplot as plot


def f(x):
    return np.sin(2 * np.pi * x)


# 样本点的数量
n = 7
# 方程的项数次数
order_ = 11

order = order_ + 1

lambd = 1e-3
epochs = 100
epsilon = 0.0000001

# 准备数据
x = np.linspace(0, 1, n)
X = np.asarray([np.power(x, i) for i in range(order)])
X = X.T
noise = np.random.normal(0, 0.2, (1, n))
Y = (f(x) + noise).T
# Y = np.asarray([[-0.5797487],
#                 [1.00653412],
#                 [0.5287666],
#                 [-0.46211614],
#                 [-0.39620244],
#                 [-0.86061157],
#                 [0.42344919]])

W = 0.1 * np.ones((order, 1))

# 开始训练
W, costs = util.conjugate_grad(W, X, Y, epochs=epochs, lambd=lambd)
# print(costs[-1])

# 画图
X_ = np.linspace(0, 1, 100)
plot.scatter(x, Y.T, label="sample")
ans = [np.sum(list(W[i] * np.power(z, i) for i in range(order))) for z in X_]
plot.title("conjugate descent, lambda = {}".format(lambd))
plot.plot(X_, ans, label="output")
plot.plot(X_, [np.sin(2 * np.pi * z) for z in X_], color="red", label="sin(2*pi*x)")
plot.show()
plot.plot(list(i + 1 for i in range(len(costs))), costs)
plot.show()
