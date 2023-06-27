import numpy as np
from numpy.linalg import norm
import pandas as pd
from sklearn.metrics import silhouette_score


class KMeans():

    def __init__(self, k=3, max_iterations=200,
                 normalize=True, distance='euclidean', 
                 weights=None, repeat=1, auto_k=True, k_range=(2,10)) -> None:
        self.k = k
        self.max_iterations = max_iterations
        self.normalize = normalize
        self.distance = distance
        self.weights = weights
        self.repeat = repeat
        self.auto_k = auto_k
        self.k_range = k_range

    def learn(self, X:pd.DataFrame):

        self.N, self.M = X.shape
        
        self.X_mean = X.mean()
        self.X_std = X.std()

        # Normalization
        if self.normalize:
            X = (X - self.X_mean) / self.X_std
        self.X = X

        # Initialize weights if none are passed
        if not self.weights:
            self.weights = np.ones(self.M)

        self.best_quality = float('inf')
        self.best_silhouette = -1

        # Find best k
        if self.auto_k:
            for k in range(self.k_range[0], self.k_range[1]+1):     
                self.k = k  

                # Repeat n times
                for r in range(self.repeat):
                    self._single_KMeans(X)

                score = self._silhouette_score(X, self.assign)
                print(f' K: {k}, quality: {self.total_quality}, silhouette: {score}')

                if score > self.best_silhouette:
                    self.best_k = k
                    self.best_auto_centroids = self.best_centroids
                    self.best_silhouette = score

            self.best_centroids = self.best_auto_centroids
        else:
            # Repeat n times
            for r in range(self.repeat):
                self._single_KMeans(X)

                print(f' Iteration: {r+1}, quality: {self.total_quality}')

    def _single_KMeans(self, X):

        # Initialization
        X_n = X.to_numpy()
        centroids = self._init_centroids(X)
        self.assign = np.zeros((self.N,1))
        old_quality = float('inf')

        for iteration in range(self.max_iterations):
            quality = np.zeros(self.k)

            # Assign centroids to data points
            for i in range(self.N):
                instance = X_n[i]
                dist = self._calculate_distance(instance, centroids)
                self.assign[i] = np.argmin(dist)

            # Recalculate centroids
            for c in range(self.k):
                subset = X[self.assign==c]
                centroids[c] = subset.mean()
                quality[c] = subset.var().sum() * len(subset) # SSE

            self.total_quality = quality.sum()
            # print(iteration, self.total_quality)

            # Break if converges
            if old_quality == self.total_quality: break

            old_quality = self.total_quality

        if self.total_quality < self.best_quality:
            self.best_quality = self.total_quality 
            self.best_centroids = centroids

    def transform(self, X):

        N, M = X.shape
        if self.normalize:
            X = (X - X.mean()) / X.std()
        X = X.to_numpy()

        
        for i in range(N):
            instance = X[i]
            dist = self._calculate_distance(instance, self.best_centroids)
            self.assign[i] = np.argmin(dist)
        return self.assign

    def _calculate_distance(self, instance, centroids):
        if self.distance == 'euclidean':
            dist = np.sqrt((instance-centroids)**2)
        elif self.distance == 'absolute':
            dist = (np.abs(instance-centroids))
        elif self.distance == 'cosine':
            for i in range(len(centroids)):
                dist = np.zeros(len(centroids))
                dist[i] = (np.dot(instance, centroids[i])/(norm(instance)*norm(centroids)))
        elif self.distance == 'canberra':
            dist = (np.abs(instance-centroids)/(np.abs(instance)+np.abs(centroids)))

        return (dist * self.weights).sum(axis=1)
        
    def _init_centroids(self, X):

        X = X.to_numpy()
        centroids = np.empty((self.k, self.M))
        
        # Select random datapoint as first centroid
        first_centroid_idx = np.random.choice(range(self.N))
        centroids[0] = X[first_centroid_idx]

        # Calculate distances
        distances = np.linalg.norm(X - centroids[0], axis=1)
        
        for i in range (1,self.k):

            # Calculate probabilities and chose the next centroid 
            # with probability proportional to distance squared
            probabilities = distances ** 2 / np.sum(distances ** 2)
            next_centroid_idx = np.random.choice(range(self.N), p=probabilities)
            centroids[i] = X[next_centroid_idx]
            
            # Update the distances using the new centroid
            new_distances = np.linalg.norm(X - centroids[i], axis=1)
            distances = np.minimum(distances, new_distances)

        return centroids
    
    def get_cluster_quality(self):

        # De-normalize centroids
        centroids = pd.DataFrame(self.best_centroids, columns=self.X.columns)
        centroids = centroids * self.X_std + self.X_mean

        print('\nCentroids:')
        print(centroids)

        # Calculate uniqueness of clusters
        print('\nDistances between clusters:')
        for i in range(len(self.best_centroids)):
            for j in range(i+1,len(self.best_centroids)):
                distance = ((self.best_centroids[i]-self.best_centroids[j])**2).sum()
                print(f'SSE for cluster {i} and {j} is {distance:.2f}')

        # Calculate cluster cohesion
        print('\nCluster cohesion:')
        for c in range(len(self.best_centroids)):
            subset = self.X[self.assign==c]
            total_error = subset.var().sum() * len(subset)

            print(f'Total error for cluster {c} is {total_error:.2f}')

        print('\nValue counts:')
        self.X['clusters'] = self.assign
        print(self.X.clusters.value_counts())

    def _silhouette_score_old(self, X):
        X_n = X.to_numpy()
        scores = []

        # calculate for single data
        for i in range(self.N):
            inter_cluster_distance = float('inf')
            for c in range(len(self.best_centroids)):
                distance = np.sqrt((X_n[i]-self.best_centroids[c])**2).sum()
                if c != self.assign[i] and distance < inter_cluster_distance:
                    inter_cluster_distance = distance

            intra_cluster_distance = 0
            instances = X[self.assign == self.assign[i]].to_numpy()

            for instance in instances:
                intra_cluster_distance = np.sqrt((X_n[i]-instance)**2).sum()

            intra_cluster_distance /= len(instances)

            denominator = max(inter_cluster_distance, intra_cluster_distance)
            dif = (inter_cluster_distance - intra_cluster_distance)
            
            if denominator != 0:
                scores.append(dif / denominator)
            
        #calculate mean
        final_score = np.mean(scores)
        return final_score
    
    def _silhouette_score(self, X, labels):
        X = X.to_numpy()
        n_samples = X.shape[0]
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters == 1:
            return 0.0
        
        distances = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
        
        a_values = np.zeros(n_samples)
        b_values = np.zeros(n_samples)
        
        for i in range(n_samples):
            mask = labels == labels[i]
            a_values[i] = np.mean(distances[i, mask])
            
            other_clusters = np.unique(labels[~mask])
            b_values[i] = np.min([np.mean(distances[i, labels == c]) for c in other_clusters])
        
        silhouette_values = (b_values - a_values) / np.maximum(a_values, b_values)
        average_silhouette_score = np.mean(silhouette_values)
        
        return average_silhouette_score


if __name__ == '__main__':

    df = pd.read_csv('data/boston.csv').drop('MEDV', axis=1)

    model = KMeans(k=3, repeat=2, auto_k=False)
    model.learn(df)
    clusters = model.transform(df)
    model.get_cluster_quality()
