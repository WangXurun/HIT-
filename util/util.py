import numpy as np
import h5py


def grad(X, Y, W, lambd=0):
    return np.dot(np.asarray(X).T, np.dot(X, W) - np.asarray(Y)) + lambd * W


def decent(W, alpha, grad):
    return W - alpha * grad


def SSE(Y, Y_pred):
    return np.sum(0.5 * np.square(Y_pred - Y))


def cost_with_regular(Y, Y_pred, lambd, W):
    return np.sum(0.5 * np.square(Y_pred - Y) + 0.5 * lambd * (1 / np.dot(W.T, W)))


def conjugate_grad(W, X, Y, epsilon=1e-3, epochs=1000000, lambd=0):
    X_ = np.dot(X.T, X) + lambd * np.eye(X.shape[1])
    r = np.dot(X.T, (Y - np.dot(X, W)))
    p = r
    k = 0
    costs = []
    while k < epochs:
        # print(k)
        alpha = np.dot(r.T, r) / np.dot(np.dot(p.T, X_), p)
        # print(alpha)
        W = W + alpha[0][0] * p
        this_r = r - np.dot(alpha * X_, p)
        cost_ = SSE(Y, np.dot(X, W))
        costs.append(cost_)
        if np.all(cost_ < epsilon):
            return W, costs
        k += 1
        beta = np.dot(this_r.T, this_r) / np.dot(r.T, r)
        r = this_r
        p = r + beta * p
    return W, costs


def analytical(X, Y, lambd):
    return np.linalg.inv(np.dot(X.T, X) + lambd * np.eye(X.shape[1])).dot(X.T).dot(Y)


def cost(y, y_pred):
    return np.sum(y * np.log(y_pred))


def CrossEntropy(y, y_pred):
    return -1 * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def logistic_forward(W, X):
    return sigmoid(np.dot(W, X))


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def logistic_grad(X, Y, W, lambd=0, random=False):
    if random:
        i = np.random.randint(0, X.shape[1])
        p = X.T
        p = p[i]
        return p * np.squeeze((logistic_forward(W, X) - Y))[i] + (lambd * W) / X.shape[1]
    # print(a.shape)
    return np.dot(logistic_forward(W, X) - Y, X.T) / X.shape[1] + (lambd * W) / X.shape[1]


def norm(x, axis=0, mu=None, max_=None, min_=None):
    if axis == 0:
        shape = (1, -1)
    else:
        shape = (-1, 1)
    if mu is not None:
        return ((x - mu) / (max_ - min_))
    mu = np.mean(x, axis=axis).reshape(shape)
    max_ = np.max(x, axis=axis).reshape(shape)
    min_ = np.min(x, axis=axis).reshape(shape)
    return ((x - mu) / (max_ - min_)), mu, max_, min_


def load_data(path):
    x = np.loadtxt(path, dtype=np.float, delimiter=",")
    X = x[:, :-1].tolist()
    Y = x[:, -1].tolist()
    return X, Y


def EM_E(k, X, mu, pi, sigma):
    n = len(X)
    gamma = np.zeros((n, k))
    for i in range(0, n):
        tmp = 0
        for j in range(0, k):
            tmp += pi[j] * norm_2(X[i], mu[j], sigma[j])
        for j in range(0, k):
            gamma[i, j] = pi[j] * norm_2(X[i], mu[j], sigma[j]) / tmp
    labels = np.argmax(gamma, axis=1)
    return gamma, labels


def EM_M(num_of_k, X, gamma):
    n = len(X)

    sigma = np.zeros((num_of_k, 2, 2))
    # u = np.zeros((num_of_k, 2))
    N = np.sum(gamma, axis=0)

    total = np.dot(gamma.T, X)

    mu = total / N.reshape(num_of_k, 1)
    for i in range(0, num_of_k):
        s = np.zeros((2, 2))
        for j in range(0, n):
            temp = np.asmatrix([X[j] - mu[i]])
            s += gamma[j, i] * temp.T * temp
        sigma[i] = (s / N[i]).tolist()
    pi = (N / n).tolist()
    return mu, sigma, pi


def norm_2(X, mu, sigma):
    sigma = np.asarray(sigma, dtype=np.float32)
    return 1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma))) * np.exp(
        -0.5 * np.dot((X - mu).T, np.dot(np.linalg.inv(sigma), (X - mu))))


def EM_lnp(num_of_k, X, mu, pi, sigma):
    n = len(X)
    ans = 0
    for i in range(n):
        sum = 0
        for j in range(num_of_k):
            sum += pi[j] * norm_2(X[i], mu[j], sigma[j])
        ans += np.log(sum)
    return ans
