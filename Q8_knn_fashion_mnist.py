import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
X = data.data[:10000] / 255.0
y = data.target[:10000].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for k in [3, 5, 7]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("K =", k)
    print("Accuracy:", accuracy_score(y_test, pred))

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
pred = model.predict(X_test)

plt.imshow(X_test[0].reshape(28, 28), cmap="gray")
plt.title("Actual: " + str(y_test[0]) + " Predicted: " + str(pred[0]))
plt.show()
