from scipy.spatial.distance import cdist 
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import random

# numpix,colors = (36,3)
# K = 3
# iters = 3
# # 6x6 pixel photo, 3 color density
# # set each pixel itensity value to a random value 
# # make our input matrix

def getDist(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def assignCenters(X,centroids, clusters, K):
    # assign the closest center to a sample to create clusters
    clusters = [[] for _ in range(K)]
    for idx, sample in enumerate(X):
        centroid_idx = calcCenters(sample, centroids)
        clusters[centroid_idx].append(idx)
    return clusters

def calcCenters( sample, centroids):
    #calc distances from each sample to each center
    distances = [getDist(sample, point) for point in centroids]
    closest_index = np.argmin(distances)
    return closest_index

def getIDS( clusters,numpix):
    # each sample will get the label of the cluster it was assigned to
    labels = np.empty(numpix)
    for cluster_idx, cluster in enumerate(clusters):
        for sample_index in cluster:
            labels[sample_index] = cluster_idx
    return labels


def _is_converged( centroids_old, centroids,K):
    # distances between each old and new centroids, fol all centroids
    distances = [getDist(centroids_old[i], centroids[i]) for i in range(K)]
    return sum(distances) == 0


def plot(X,clusters,centroids):
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, index in enumerate(clusters):
        point = X[index].T
        ax.scatter(*point)
    for point in centroids:
        ax.scatter(*point, marker="x", color='black', linewidth=2)
    plt.show()
    
np.random.seed(42)
numpix,colors = (16,3)
K = 3
iters = 5
steps = True
X = [random.sample(range(0,255),3) for b in range(numpix)]


# list of sample indices for each cluster
clusters = [[] for _ in range(K)]
# the centers (mean feature vector) for each cluster
centroids = []


# initialize 
means = np.random.choice(numpix, K, replace= False)
centroids = [X[idx] for idx in means]
# Optimize clusters
for _ in range(iters):
    # Assign samples to closest centroids (create clusters)
    clusters = assignCenters(X,centroids, clusters, K)
    if steps:
        plot(X,clusters,centroids)
    # Calculate new centroids from the clusters
    centroids_old = centroids
    centroids = calcCenters(clusters, centroids_old)
    
    # check if clusters have changed
    if _is_converged(centroids_old, centroids,K):
        break
    if steps:
        plot()
# Classify samples as the index of their clusters
ids =  getIDS(clusters, numpix)

print(ids)

