import numpy as np
import matplotlib.pyplot as plot
import mpl_toolkits.mplot3d.axes3d as axes3d
import cv2
import util.util as util

# 生成数据
np.random.seed(2)
alpha = np.pi / 6
beta = np.pi / 5
gamma = np.pi / 10
# 旋转矩阵
A = [[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)]]
B = [[np.cos(beta), 0, -np.sin(beta)], [0, 1, 1], [np.sin(beta), 0, np.cos(beta)]]
C = [[np.cos(gamma), np.sin(gamma), 0], [-np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]]

mu = [1, 1, 3]
sigma = [[2, 0, 0], [0, 2, 0], [0, 0, 0.1]]
X = np.random.multivariate_normal(mu, sigma, 100)
# print(np.max(X[:, 0]), np.max(X[:, 1]), np.max(X[:, 2]))
# print(np.min(X[:, 0]), np.min(X[:, 1]), np.min(X[:, 2]))
X = X.dot(A).dot(B).dot(C)
print(np.max(X[:, 0]), np.max(X[:, 1]), np.max(X[:, 2]))
print(np.min(X[:, 0]), np.min(X[:, 1]), np.min(X[:, 2]))

fig = plot.figure()
ax = fig.gca(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], label="original", color="blue")  # 显示生成的数据
ax.set_xlim(-4, 9)
ax.set_ylim(-4, 9)
ax.set_zlim(-4, 9)
# plot.show()
n, m = X.shape

# 设置降低到几维
k = 2
l, u, mean = util.PCA(X, 2)
Z = np.dot(u, (X - mean).T)
out = np.dot(Z.T, np.asarray(u)) + mean  # 数据重建
ax.scatter(out[:, 0], out[:, 1], out[:, 2], label="reconstructed", color="red")

# plot.plot(Z[0], Z[1], ".")
plot.legend()
plot.show()

# lena 图的降维
rate = 0.5  # 数据保留比例
img = cv2.imread("dataset/lena_grey.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
la, u_, mean_ = util.PCA(img,int(len(img) * rate))
z = np.dot(u_, np.asarray(img - mean_).T)
# cv2.imshow("1",z.T)
# cv2.waitKey(25)
out = np.dot(z.T, np.asarray(u_)) + mean_
out = out.astype(np.uint8)
a = util.psnr(img, out)
print(a)
cv2.imshow(f"", out)
cv2.waitKey(25)

# 绘制psnr随k值的变化曲线
# ls=[]
# index=[]
# for i in range(1,len(img)+1):
#
#     la, u_, mean_ = util.PCA(img, i)
#     z = np.dot(u_, np.asarray(img - mean_).T)
#     # cv2.imshow("1",z.T)
#     # cv2.waitKey(25)
#     out = np.dot(z.T, np.asarray(u_)) + mean_
#     out = out.astype(np.uint8)
#     a=util.psnr(img, out)
#     # print(a)
#     ls.append(a)
#     index.append(i)
# plot.plot(index,ls)
# plot.xlabel("k")
# plot.ylabel("PNSR")
# plot.legend()
# plot.show()

# cv2.imshow(f"", out)
# cv2.waitKey(25)
