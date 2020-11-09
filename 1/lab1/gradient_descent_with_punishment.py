import numpy as np
import util.util as util
import matplotlib.pyplot as plot


def f(x):
    return np.sin(2 * np.pi * x)


# 样本点的数量
n = 7
# 方程的次数
order_ = 11

order = order_ + 1


lambd = 1e-3
epochs = 5000

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

W = 0.1 * np.ones((order, 1))
last_cost = 0
costs = []

#开始训练
for epoch in range(epochs):
    Y_pred = np.dot(X, W)
    grad = util.grad(X, Y, W, lambd=lambd)
    alpha_ = np.dot(grad.T, grad) / np.dot((np.dot(grad.T, np.dot(X.T, X) + lambd * np.eye(X.shape[1]))), grad)
    W = util.decent(W, alpha_, grad)
    cost = util.SSE(Y, Y_pred)
    costs.append(cost)
    if (epoch + 1) % 10000 == 0:
        print("epoc : {} , cost : {} , alpha : {}".format(epoch + 1, cost, alpha_))

# 画图
title = "gradient descent with punishment,m={} , n={} , lambda={} ".format(order - 1, n, lambd)
X_ = np.linspace(0, 1, 100)
plot.title(title)
plot.scatter(x, Y.T, label="sample")
ans = [np.sum(list(W[i] * np.power(z, i) for i in range(order))) for z in X_]
plot.plot(X_, ans, label="output")
plot.plot(X_, [np.sin(2 * np.pi * z) for z in X_], color="red", label="sin(2*pi*x)")
plot.legend()
plot.show()
plot.xlabel("epoch")
plot.ylabel("losses")
plot.title(title)
plot.plot(list(i + 1 for i in range(len(costs))), costs, label="cost")
plot.show()
