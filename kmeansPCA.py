import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import csvRead
import ast
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
X = []
vecs = csvRead.fealook2()
count=1
for y in vecs:
        res = ast.literal_eval(y)
        X.append(res)
X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2).fit(X)
datapoint = pca.transform(X)

kmeans_model = KMeans(n_clusters=5, init='k-means++', n_init=2000, max_iter=6000, precompute_distances='auto')
x = kmeans_model.fit(datapoint)
labels = kmeans_model.labels_.tolist()
clusters = kmeans_model.fit_predict(datapoint)
centroids = kmeans_model.cluster_centers_

print(metrics.silhouette_score(datapoint, labels, metric='euclidean'))
print(metrics.calinski_harabasz_score(datapoint, labels))
print(metrics.davies_bouldin_score(datapoint,labels))
print(metrics.silhouette_samples(datapoint,labels))

plt.figure
label1 = ["#FFFF00", "#008000", "#0000FF", "#800080", "#FF0000","#8f9805","#ffde57","#4584b6","#646464"]
color = [label1[i] for i in labels]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='o',
            c="white", alpha=1, s=200, edgecolor='k')

for i, c in enumerate(centroids):
    plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                s=50, edgecolor='k')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='^', s=150, c='#000000')
plt.show()


"""
for i, k in enumerate([4,5,6,7,8]):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # Run the Kmeans algorithm
    km = KMeans(n_clusters=k)
    labels = km.fit_predict(datapoint)
    centroids = km.cluster_centers_

    # Get silhouette samples
    silhouette_vals = silhouette_samples(datapoint, labels)

    # Silhouette plot
    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)

    # Get the average silhouette score and plot it
    avg_score =metrics.silhouette_score(datapoint, labels, metric='euclidean')
    print(metrics.silhouette_score(datapoint, labels, metric='euclidean'))
    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
    ax1.set_yticks([])
    ax1.set_xlim([-0.1, 1])
    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster labels')
    ax1.set_title('Silhouette plot for the various clusters', y=1.02);

    # Scatter plot of data colored with labels
    ax2.scatter(datapoint[:, 0], datapoint[:, 1], c=labels)
    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r', s=250)
    ax2.set_xlim([-2, 2])
    ax2.set_xlim([-2, 2])
    ax2.set_xlabel('Eruption time in mins')
    ax2.set_ylabel('Waiting time to next eruption')
    ax2.set_title('Visualization of clustered data', y=1.02)
    ax2.set_aspect('equal')
    plt.tight_layout()
    plt.suptitle(f'Silhouette analysis using k = {k}',fontsize=16, fontweight='semibold', y=1.05);
plt.show()

from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=5).fit(datapoint)
labels = clustering.labels_.tolist()
clusters = clustering.fit_predict(datapoint)


print(metrics.silhouette_score(datapoint, labels, metric='euclidean'))

plt.figure
label1 = ["#FFFF00", "#008000", "#0000FF", "#800080", "#FF0000"]
color = [label1[i] for i in labels]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)
plt.show()

for i, k in enumerate([5,6,7,8]):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # Run the Kmeans algorithm
    km = KMeans(n_clusters=k)
    labels = km.fit_predict(datapoint)
    centroids = km.cluster_centers_

    # Get silhouette samples
    silhouette_vals = silhouette_samples(datapoint, labels)

    # Silhouette plot
    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)

    # Get the average silhouette score and plot it
    avg_score = np.mean(silhouette_vals)
    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
    ax1.set_yticks([])
    ax1.set_xlim([-0.1, 1])
    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster labels')
    ax1.set_title('Silhouette plot for the various clusters', y=1.02);

    # Scatter plot of data colored with labels
    ax2.scatter(datapoint[:, 0], datapoint[:, 1], c=labels)
    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r', s=250)
    ax2.set_xlim([-2, 2])
    ax2.set_xlim([-2, 2])
    ax2.set_xlabel('Eruption time in mins')
    ax2.set_ylabel('Waiting time to next eruption')
    ax2.set_title('Visualization of clustered data', y=1.02)
    ax2.set_aspect('equal')
    plt.tight_layout()
    plt.suptitle(f'Silhouette analysis using k = {k}',fontsize=16, fontweight='semibold', y=1.05);
plt.show()
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(X)

db = DBSCAN(eps=1.3, min_samples=5, leaf_size=30).fit(new_values)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

unique_labels = set(labels)
colors = ['goldenrod','mediumorchid','tan','navy','maroon','forestgreen','deeppink','olive','lightcyan','royalblue','b','g', 'r', 'c', 'm', 'y']

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,markeredgecolor='k',markersize=6)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,markeredgecolor='k',markersize=6)

plt.title('number of clusters: %d' % n_clusters_)
plt.show()


db = DBSCAN(eps=0.11, min_samples=5,metric='cosine')
result = db.fit_predict(X)
svd = TruncatedSVD(n_components=2).fit_transform(X)
#Set the colour of noise pts to black
for i in range(0,len(result)):
        if result[i] == -1:
            result[i] = 7
colors = [db.labels_[l] for l in result]
plt.scatter(svd[:,0], svd[:,1], c=colors, s=50, linewidths=0.5, alpha=0.7)
plt.show()

K_value = 7
kmeans_model = KMeans(n_clusters=K_value, init='k-means++', n_init=2000, max_iter=6000, precompute_distances='auto')
kk = kmeans_model.fit(X_embedded)
labels = kmeans_model.labels_.tolist()
clusters = kmeans_model.fit_predict(X_embedded)

# PCA
l = kmeans_model.fit_predict(X_embedded)
pca = PCA(n_components=2).fit(X_embedded)
datapoint = pca.transform(X_embedded)

# GRAPH
# **Plot the clustering result**

plt.figure
label1 = ["#FFFF00", "#008000", "#0000FF", "#800080", "#FF0000","#880080","#FF8800"]
color = [label1[i] for i in labels]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
plt.show()

"""