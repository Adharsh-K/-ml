from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model1 = LinearRegression()
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)

values = [0.01, 0.1, 1, 10, 100]

model2 = GridSearchCV(Ridge(), {"alpha": values}, cv=5)
model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)

model3 = GridSearchCV(Lasso(max_iter=10000), {"alpha": values}, cv=5)
model3.fit(X_train, y_train)
pred3 = model3.predict(X_test)

print("Linear Regression")
print("MSE:", mean_squared_error(y_test, pred1))
print("R2:", r2_score(y_test, pred1))

print("Ridge Regression")
print("Best alpha:", model2.best_params_["alpha"])
print("MSE:", mean_squared_error(y_test, pred2))
print("R2:", r2_score(y_test, pred2))

print("Lasso Regression")
print("Best alpha:", model3.best_params_["alpha"])
print("MSE:", mean_squared_error(y_test, pred3))
print("R2:", r2_score(y_test, pred3))
