import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import f1_score


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


X, y = make_blobs(n_samples=400, n_features=5, centers=5, cluster_std=0.0001, random_state=20)
# iris = load_iris()
# X = iris.data
# y = iris.target

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=16, n_clusters=None, affinity='euclidean', linkage='average')
# manhattan euclidean
model = model.fit(X)

print('adjusted_rand_score: {}'.format(adjusted_rand_score(y, model.labels_)))

print('jaccard_score: {}'.format(jaccard_score(y, model.labels_, average='micro')))

print('fowlkes_mallows_score: {}'.format(fowlkes_mallows_score(y, model.labels_)))

print('f1_score: {}'.format(f1_score(y, model.labels_, average='micro')))


plt.figure('Dendrogram')
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=20)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")

plt.figure('Elements')
plt.subplot(1, 2, 1)
plt.title('True clusters')
plt.scatter(X[:, 0], X[:, 1], c=y)

plt.subplot(1, 2, 2)
plt.title('Predicted clusters')
plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
plt.show()