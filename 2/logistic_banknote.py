import numpy as np
import util.util as util
import matplotlib.pyplot as plot

# 设置参数
epochs = 100000
alpha = 2e-4
lambd = 0
title = "dataset : banknote , lambda : {}".format(lambd)

# 准备数据
train_set = np.loadtxt("dataset/banknote_train.txt", dtype=np.float32, delimiter=",").T
# train_set = (train_set.T[:2]).T
(m, n) = train_set.shape
m = m - 1

X = train_set[:-1]
Y = train_set[-1]
Y.reshape((1, n))
X = np.vstack((np.ones((1, n)), X))
costs = []

# 准备测试集
test_set = np.loadtxt("dataset/banknote_test.txt", dtype=np.float32, delimiter=",").T
X_test = test_set[:-1]
Y_test = test_set[-1]
X_test = np.vstack((np.ones((1, X_test.shape[1])), X_test))

# 初始化模型参数
W = np.ones((1, m + 1)) * 0.1
ac = []
last_cost = 0

# 开始训练
for epoch in range(epochs):
    Y_pred = util.logistic_forward(W, X)
    cost = util.CrossEntropy(Y, Y_pred) / len(Y)
    costs.append(cost)
    if float(cost) > last_cost:
        alpha = 0.5 * alpha
        last_cost = cost
    grad = util.logistic_grad(X, Y, W, lambd=lambd, random=False)
    W -= alpha * grad
    Y_pred = util.logistic_forward(W, X_test)
    mask = (Y_pred >= 0.5) == Y_test
    mask = np.squeeze(mask)
    ac.append(np.sum(mask) / len(mask))
    # print("accuracy:{}".format(np.sum(mask) / len(mask)))
    if (epoch + 1) % 1000 == 0:
        print("epoch : {}  loss : {} alpha : {}".format(epoch + 1, cost, alpha))

# 尝试动量梯度
# beta = 0.9
# v = np.zeros(W.shape)
# for epoch in range(epochs):
#     Y_pred = util.logistic_forward(W, X)
#     cost = util.CrossEntropy(Y, Y_pred) / len(Y)
#     costs.append(cost)
#     if float(cost) > last_cost:
#         alpha = 0.5 * alpha
#         last_cost = cost
#     grad = util.logistic_grad(X, Y, W, lambd=lambd, random=False)
#     v = beta * v + (1 - beta) * grad
#     W -= alpha * v
#     Y_pred = util.logistic_forward(W, X_test)
#     mask = (Y_pred >= 0.5) == Y_test
#     mask = np.squeeze(mask)
#     ac.append(np.sum(mask) / len(mask))
#     if (epoch + 1) % 1000 == 0:
#         print("epoch : {}  loss : {} alpha : {}".format(epoch + 1, cost, alpha))
print(np.max(ac))
print(ac[-1])

# 画图
plot.plot([i + 1 for i in range(len(ac))], ac, color="red", label="accuracy")
plot.xlabel("epoch")
plot.ylabel("accuracy")
plot.title(title)
plot.legend()
plot.show()
plot.plot([i + 1 for i in range(len(costs))], costs, label="loss")
plot.xlabel("epoch")
plot.ylabel("loss")
plot.title(title)
plot.legend()
plot.legend()
plot.show()

# 计算查准率和召回率
Yp = util.logistic_forward(W, X_test) >= 0.5
mask = Yp == Y_test
print("查准率：{}".format(np.sum(mask * Y_test) / np.sum(Yp)))
print("召回率：{}".format(np.sum(mask * Y_test) / np.sum(Y_test)))

# plotBD(W, "green")
# plot.show()
