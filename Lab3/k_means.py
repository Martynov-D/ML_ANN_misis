from sklearn.cluster import k_means

from matplotlib import pyplot as plt

from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import f1_score


X, y = make_blobs(n_samples=400, n_features=2, centers=5, cluster_std=3.5, random_state=20)
# iris = load_iris()
# X = iris.data
# y = iris.target

results = k_means(X, 5)


print('adjusted_rand_score: {}'.format(adjusted_rand_score(y, results[1])))

print('jaccard_score: {}'.format(jaccard_score(y, results[1], average='micro')))

print('fowlkes_mallows_score: {}'.format(fowlkes_mallows_score(y, results[1])))

print('f1_score: {}'.format(f1_score(y, results[1], average='micro')))

plt.figure('Elements')
plt.subplot(1, 2, 1)
plt.title('True clusters')
plt.scatter(X[:, 0], X[:, 1], c=y)

plt.subplot(1, 2, 2)
plt.title('Predicted clusters')
plt.scatter(X[:, 0], X[:, 1], c=results[1])
plt.show()