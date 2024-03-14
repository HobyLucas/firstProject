import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

x, y = make_regression(n_samples=100, n_features=1, noise=10)

y = y.reshape(y.shape[0], 1)

print(x.shape)
print(y.shape)


#matrice X
X = np.concatenate((x, np.ones(x.shape)), axis=1)
print(X)

#vecteur theta
theta = np.random.randn(2, 1)
print(theta)

#creation du modele
def modele(X, theta):
    return X.dot(theta)


def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((modele(X, theta) - y)**2)

def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(modele(X, theta) - y)

def gradient_descent(X, y, theta, learning_rate, n_iteration):

    for i in range(0, n_iteration):
        theta = theta - learning_rate * grad(X, y, theta)

    return theta

theta_final = gradient_descent(X, y, theta, learning_rate=0.01, n_iteration=1000)

prediction = modele(X, theta_final)

plt.scatter(x, y)
plt.plot(x, prediction, c='r')
plt.show()

def coef_det(y, pred):
    u = ((y - pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - u/v

print(coef_det(y, prediction))