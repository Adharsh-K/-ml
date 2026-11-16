import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

columns = ["mpg", "cylinders", "displacement", "horsepower", "weight",
           "acceleration", "year", "origin", "name"]

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
                 sep=r"\\s+", names=columns, na_values="?")

df = df[["displacement", "mpg"]]
df["displacement"] = pd.to_numeric(df["displacement"], errors="coerce")
df["mpg"] = pd.to_numeric(df["mpg"], errors="coerce")
df = df.dropna()

X = df[["displacement"]].values
y = df["mpg"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Linear Regression")
print("MSE:", mean_squared_error(y_test, pred))
print("R2:", r2_score(y_test, pred))

for d in [2, 3, 4]:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    pred = model.predict(X_test_poly)

    print("Degree:", d)
    print("MSE:", mean_squared_error(y_test, pred))
    print("R2:", r2_score(y_test, pred))

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(poly.transform(x_line))

plt.scatter(X, y)
plt.plot(x_line, y_line, color="red")
plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.show()
