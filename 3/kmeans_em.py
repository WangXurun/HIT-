import numpy as np
import matplotlib.pyplot as plot
import util.util as util
import sklearn.metrics as metrics

markersize = 20
num_of_k = 4


def gen_data():
    mu1 = np.asarray([1, 1])
    mu2 = np.asarray([2, 2])
    mu3 = np.asarray([1, 2])
    mu4 = np.asarray([2, 1])
    sigma1 = np.asarray([[0.08, 0], [0, 0.08]])
    sigma2 = np.asarray([[0.08, 0.02], [0.02, 0.08]])
    sigma3 = np.asarray([[0.08, 0.04], [0.04, 0.08]])
    # sigma2 = np.asarray([[1, 0], [0, 1]])
    s1 = np.random.multivariate_normal(mu1, sigma1, 300)
    s2 = np.random.multivariate_normal(mu2, sigma2, 300)
    s3 = np.random.multivariate_normal(mu3, sigma3, 300)
    s4 = np.random.multivariate_normal(mu4, sigma2, 300)
    plot.plot(*s1.T, '.', color='red')
    plot.plot(*s2.T, '.', color='blue')
    plot.plot(*s3.T, '.', color='green')
    plot.plot(*s4.T, '.', color='orange')
    plot.title("original data")
    plot.show()
    x1 = s1[:, 0]
    x2 = s1[:, 1]
    x3 = s2[:, 0]
    x4 = s2[:, 1]
    x5 = s3[:, 0]
    x6 = s3[:, 1]
    x7 = s4[:, 0]
    x8 = s4[:, 1]
    with open("D:\\学习文件\\机器学习\\lab\\3\\dataset\\4-cluster_train.txt", 'w') as f:
        for i in range(len(x1)):
            f.write("{},{},0\n".format(x1[i], x2[i]))
            f.write("{},{},1\n".format(x3[i], x4[i]))
            f.write("{},{},2\n".format(x5[i], x6[i]))
            f.write("{},{},3\n".format(x7[i], x8[i]))


def kmeans(dotes, Y):
    changed = True
    labels = [0] * len(dotes)
    centers = np.random.rand(num_of_k, 2) *3
    classes = [[], [], [], []]
    # classes = [[], [], []]
    cnt = 0
    ls = []
    while changed:
        cnt += 1
        classes = [[], [], [], []]
        # classes = [[], [], []]
        changed = False
        for i in range(len(dotes)):
            x_, y_ = dotes[i][0], dotes[i][1]
            min = 100
            index = 0
            for j in range(centers.shape[0]):
                x = centers[j][0]
                y = centers[j][1]
                dis = (x - x_) ** 2 + (y - y_) ** 2
                if dis < min:
                    index = j
                    min = dis
            if labels[i] != index:
                changed = True
                labels[i] = index
        for i in range(len(dotes)):
            tmp = classes[labels[i]]
            tmp.append(dotes[i])
        for i in range(len(classes)):
            c = classes[i]
            if len(c) != 0:
                sum = np.sum(c, axis=0)
                centers[i] = (sum / len(c)).tolist()
        r = metrics.adjusted_rand_score(Y, labels)
        print(f"number {cnt} epoch, rand is {r}")
        ls.append(r)
    plot.plot([i + 1 for i in range(len(ls))], ls)
    plot.show()
    return classes, centers, labels


def EM(num_of_k, X, Y, mu, sigma, pi, epoch=1):
    ls_ = []
    ls = []
    for i in range(epoch):
        gama, labels = util.EM_E(num_of_k, X, mu, pi, sigma)
        mu, sigma, pi = util.EM_M(num_of_k, X, gama)
        ll = util.EM_lnp(num_of_k, X, mu, pi, sigma)
        print(f"epoch: {i + 1} , log likelyhood: {ll}")
        ls_.append(ll)
        r = metrics.adjusted_rand_score(Y, labels)
        print(f"number {i+1} epoch, rand is {r}")
        ls.append(r)
    plot.plot([i + 1 for i in range(len(ls))], ls)
    plot.show()
    plot.plot([i + 1 for i in range(len(ls_))], ls_)
    plot.show()
    mu = np.asarray(mu)

    # plot.show()
    return labels, mu


np.random.seed(2)
gen_data()
X, Y = util.load_data("dataset/4-cluster_train.txt")
classes, centers, labels = kmeans(X, Y)
plot.title("Kmeans")
plot.plot(np.asarray(classes[0])[:, 0], np.asarray(classes[0])[:, 1], '.', color='red')
plot.plot(np.asarray(classes[1])[:, 0], np.asarray(classes[1])[:, 1], '.', color='blue')
plot.plot(np.asarray(classes[2])[:, 0], np.asarray(classes[2])[:, 1], '.', color='green')
plot.plot(np.asarray(classes[3])[:, 0], np.asarray(classes[3])[:, 1], '.', color='orange')
plot.plot(centers[0][0], centers[0][1], "+", markersize=markersize, color="black")
plot.plot(centers[1][0], centers[1][1], "+", markersize=markersize, color='black')
plot.plot(centers[2][0], centers[2][1], "+", markersize=markersize, color="black")
plot.plot(centers[3][0], centers[3][1], "+", markersize=markersize, color="black")
plot.show()
print(f"kmeans ")
print(centers)

sigma = []
for i in range(len(classes)):
    c = classes[i]
    s = np.std(c, axis=0)
    s = s.tolist()
    sigma.append([[s[0], 0], [0, s[1]]])
print(sigma)

l, m = EM(num_of_k, X, Y, centers, sigma, np.ones((num_of_k)) / num_of_k)
cl = [[], [], [], []]
# cl = [[], [], []]
for i in range(len(X)):
    tmp = cl[l[i]]
    tmp.append(X[i])
plot.plot(np.asarray(cl[0])[:, 0], np.asarray(cl[0])[:, 1], '.', color='red')
plot.plot(np.asarray(cl[1])[:, 0], np.asarray(cl[1])[:, 1], '.', color='blue')
plot.plot(np.asarray(cl[2])[:, 0], np.asarray(cl[2])[:, 1], '.', color='green')
plot.plot(np.asarray(cl[3])[:, 0], np.asarray(cl[3])[:, 1], '.', color='orange')
plot.scatter(m[:, 0], m[:, 1], marker='+', s=1000, color='black')
plot.title("EM")
plot.show()
print(metrics.adjusted_rand_score(Y, l))
