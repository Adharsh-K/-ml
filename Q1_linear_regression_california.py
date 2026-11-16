import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing()
X = data.data[:, 2].reshape(-1, 1)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mean = X_train.mean()
std = X_train.std()
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

m = 0
c = 0
lr = 0.01

for i in range(3000):
    y_pred = m * X_train_scaled[:, 0] + c
    error = y_pred - y_train
    dm = (2 / len(X_train)) * np.sum(error * X_train_scaled[:, 0])
    dc = (2 / len(X_train)) * np.sum(error)
    m = m - lr * dm
    c = c - lr * dc

pred1 = m * X_test_scaled[:, 0] + c

X_train_new = np.c_[np.ones(len(X_train)), X_train]
theta = np.linalg.inv(X_train_new.T @ X_train_new) @ X_train_new.T @ y_train
X_test_new = np.c_[np.ones(len(X_test)), X_test]
pred2 = X_test_new @ theta

print("Gradient Descent")
print("MSE:", mean_squared_error(y_test, pred1))
print("R2:", r2_score(y_test, pred1))

print("Normal Equation")
print("MSE:", mean_squared_error(y_test, pred2))
print("R2:", r2_score(y_test, pred2))

plt.scatter(X_test, y_test)
order = np.argsort(X_test[:, 0])
plt.plot(X_test[order], pred2[order], color="red")
plt.xlabel("Average Rooms")
plt.ylabel("House Price")
plt.show()
