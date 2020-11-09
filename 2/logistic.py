import numpy as np
import util.util as util
import matplotlib.pyplot as plot

# 参数设置
epochs = 100000
alpha = 2e-2
lambd = 0
# num = 500

title1 = "mu1 : (1,1)  mu2 : (2,2)  sigma : [[0.1, 0], [0, 0.1]]   lambda : {}".format(lambd)
title2 = "mu1 : (1,1)  mu2 : (2,2)  sigma : [[0.1, 0.05], [0.05, 0.1]]   lambda : {}".format(lambd)
# title3 = "mu1 : (1,1)  mu2 : (2,2)  sigma : [[1, 0], [0, 1]]   lambda : {}  \n num_of_train : {}".format(lambd, num)


# 准备训练集
train_set = np.loadtxt("dataset/logistic_train.txt", dtype=np.float, delimiter=",").T
# train_set = train_set.T[:num].T
(m, n) = train_set.shape
m = m - 1
X = train_set[:-1]
Y = train_set[-1]
Y.reshape((1, n))
X = np.vstack((np.ones((1, n)), X))
# Y = Y[:30]
costs = []
x1 = X[1]
x2 = X[2]
px1 = []
px2 = []
nx1 = []
nx2 = []
# 区分正例和反例
for i in range(len(x1)):
    if Y[i] == 1:
        px1.append(x1[i])
        px2.append(x2[i])
    else:
        nx1.append(x1[i])
        nx2.append(x2[i])
plot.title(title1)
plot.scatter(px1, px2, color="red")
plot.scatter(nx1, nx2, color="blue")

# 准备测试集
test_set = np.loadtxt("dataset/logistic_test.txt", dtype=np.float, delimiter=",").T
X_test = test_set[:-1]
Y_test = test_set[-1]
# X_test = norm(X_test, 1, mu, max_, min_)
X_test = np.vstack((np.ones((1, X_test.shape[1])), X_test))

# 初始化模型参数
W = np.ones((1, m + 1)) * 0.1


# 画分类边界
def plot_boundary(W):
    W = np.squeeze(W)
    x1 = np.arange(0, 3, step=0.1)
    # x1 = np.arange(20, 100, step=0.1)
    x2 = -(W[0] + W[1] * x1) / W[2]
    plot.plot(x1, x2, color="green")
    plot.legend(loc=3)


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
    if (epoch + 1) % 1000 == 0:
        print("epoch : {}  loss : {} alpha : {}".format(epoch + 1, cost, alpha))


# 画图
plot.title('The Decision Boundary')
plot.xlabel("x1")
plot.ylabel("x2")
plot_boundary(W)
plot.legend()
plot.show()
print(np.max(ac))
print(ac[-1])

plot.plot([i + 1 for i in range(len(ac))], ac, color="red", label="accuracy")
plot.xlabel("epoch")
plot.ylabel("accuracy")
plot.title(title1)
plot.legend()
plot.show()
plot.plot([i + 1 for i in range(len(costs))], costs, label="loss")
plot.xlabel("epoch")
plot.ylabel("loss")
plot.title(title1)
plot.legend()
plot.legend()
plot.show()

# 计算查准率和召回率

Yp = util.logistic_forward(W, X_test) >= 0.5
mask = Yp == Y_test
print("查准率：{}".format(np.sum(mask * Y_test) / np.sum(Yp)))
print("召回率：{}".format(np.sum(mask * Y_test) / np.sum(Y_test)))

# 下面是生成数据的代码
# mu1 = np.asarray([1, 1])
# mu2 = np.asarray([2, 2])
# sigma1 = np.asarray([[1, 0], [0, 1]])
# sigma2 = np.asarray([[1, 0], [0, 1]])
# s = np.random.multivariate_normal(mu1, sigma2, 500)
# s2=np.random.multivariate_normal(mu2, sigma1, 500)
# plot.plot(*s.T, '.',color='r')
# plot.plot(*s2.T, '.',color='b')
# plot.show()
# px1=s[:,0]
# px2=s[:,1]
# nx1=s2[:,0]
# nx2=s2[:,1]
# with open("D:\\学习文件\\机器学习\\lab\\2\\dataset\\overfitting_train.txt",'w') as f:
#     for i in range(int(0.7*len(px1))):
#         f.write("{},{},0\n".format(px1[i],px2[i]))
#         f.write("{},{},1\n".format(nx1[i],nx2[i]))
# with open("D:\\学习文件\\机器学习\\lab\\2\\dataset\\overfitting_test.txt",'w') as f:
#     for i in range(int(0.7*len(px1)),len(px1)):
#         f.write("{},{},0\n".format(px1[i],px2[i]))
#         f.write("{},{},1\n".format(nx1[i],nx2[i]))
