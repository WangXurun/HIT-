import numpy as np
import matplotlib.pyplot as plot
import mpl_toolkits.mplot3d.axes3d as axes3d
import cv2


def PCA(X, k):
    mean = np.mean(X, axis=0)
    X_ = X - mean
    S = np.dot(X_.T, X_) / n
    lambd, u = np.linalg.eig(S)
    a, b, c = np.linalg.svd(S)
    if len(b) < len(S):
        b = b.tolist() + [0] * (len(S) - len(b))
    return b[:k], c[:k], mean
    # u = u.T
    # la = []
    # u_ = []
    # for i in range(k):
    #     l = np.argmax(lambd)
    #     u_.append(u[l].tolist())
    #     la.append(lambd[l])
    #     lambd = np.delete(lambd, l)
    #     u = np.delete(u, l, axis=0)
    # return la, u_, mean


alpha = np.pi / 6
beta = np.pi / 3
gamma = np.pi / 10
A = [[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)]]
B = [[np.cos(beta), 0, -np.sin(beta)], [0, 1, 1], [np.sin(beta), 0, np.cos(beta)]]
C = [[np.cos(gamma), np.sin(gamma), 0], [-np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]]

mu = [1, 1, 3]
sigma = [[2, 0, 0], [0, 2, 0], [0, 0, 0.01]]
X = np.random.multivariate_normal(mu, sigma, 100)
X = X.dot(A).dot(B).dot(C)
print(np.max(X[:, 0]), np.max(X[:, 1]), np.max(X[:, 2]))
print(np.min(X[:, 0]), np.min(X[:, 1]), np.min(X[:, 2]))

fig = plot.figure()
ax = fig.gca(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)
# ax.legend()
# plot.subplot(121)
plot.show()
plot.cla()
n, m = X.shape

# 设置降低到几维
k = 2

l, u, mean = PCA(X, 2)
Z = np.dot(u, X.T)
# a = plot.figure()
# a = a.gca(projection="2d")
# a.plot(Z[0], Z[1], ".")
#
# a.show()
# plot.subplot(122)
plot.plot(Z[0], Z[1], ".")
plot.show()

img = cv2.imread("dataset/lena_grey.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("haha", img)
# cv2.waitKey(25)
la, u_, mean_ = PCA(img, int(img.shape[0] * 0.5))
z = np.dot(u_, np.asarray(img - mean_).T)
out = np.dot(z.T, np.asarray(u_)) + mean_
# print(out)
out = out.astype(np.uint8)

cv2.imshow("haha", out)
cv2.waitKey(25)

# print(l)
# print(u)
