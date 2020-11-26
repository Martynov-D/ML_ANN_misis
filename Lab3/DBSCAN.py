from sklearn.cluster import DBSCAN

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import f1_score

# X, y = make_blobs(n_samples=400, n_features=2, centers=5, cluster_std=3.5, random_state=20)
X, y = make_moons(n_samples=750, noise=0.04, random_state=20)
# iris = load_iris()
# X = iris.data
# y = iris.target

model = DBSCAN(metric='euclidean', eps=0.4, min_samples=20)
model.fit(X)

clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
noise = list(model.labels_).count(-1)

print('Clusters: {}'.format(clusters))
print('Noise points: {}'.format(noise))

print('adjusted_rand_score: {}'.format(adjusted_rand_score(y, model.labels_)))

print('jaccard_score: {}'.format(jaccard_score(y, model.labels_, average='micro')))

print('fowlkes_mallows_score: {}'.format(fowlkes_mallows_score(y, model.labels_)))

print('f1_score: {}'.format(f1_score(y, model.labels_, average='micro')))

plt.figure('Elements')
plt.subplot(1, 2, 1)
plt.title('True clusters')
plt.scatter(X[:, 0], X[:, 1], c=y)

plt.subplot(1, 2, 2)
plt.title('Predicted clusters')
plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
plt.show()
