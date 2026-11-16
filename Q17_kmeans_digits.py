import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = load_digits()
X = data.data

ks = [8, 10, 12]
inertia = []
silhouette = []

for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    inertia.append(model.inertia_)
    silhouette.append(silhouette_score(X, labels))

    print("K =", k)
    print("Inertia:", model.inertia_)
    print("Silhouette Score:", silhouette_score(X, labels))

plt.plot(ks, inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

plt.plot(ks, silhouette, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()
