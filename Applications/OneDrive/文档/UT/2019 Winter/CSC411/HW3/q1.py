import numpy as np


def gradient_descent(X, y, lr, num_iter, delta):
    w = np.zeros((len(X[1]), 1))
    b = 10
    for i in range(num_iter):
        y_predict = np.dot(X, w) + b
        alpha = y_predict - y
        dloss = np.where(np.abs(alpha) <= delta, alpha, delta * np.sign(alpha))
        dw = np.dot(np.transpose(X), dloss)
        db = np.mean(dloss)
        w = w - lr * dw
        b = b - lr * db
    return w, b


N = 10
D = 5
X = np.zeros((N, D))
y = np.zeros((N, 1))
delta = 1.
lr = 1
dw, db = gradient_descent(X, y, lr, 1000, delta)
print(dw)
print(db)
