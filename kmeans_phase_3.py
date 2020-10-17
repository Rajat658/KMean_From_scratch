# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:33:47 2020

@author: rajat
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import numpy.matlib
from sklearn.datasets import make_blobs
import altair as alt
import timeit
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#the below imported library will be used only for calculating k value
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class K_Means:
    def __init__(self, n_cluster,Max_iter=500,random_state=42,tolerance=0.0001):
        self.n_cluster=n_cluster
        self.Max_iter=Max_iter
        self.tolerance=tolerance
        self.random_state=random_state
    def fit(self,d,dataset):
        self.centroids={}
        self.k=int(input("No_of_clusters:"))
        for i in range(self.k):
            self.centroids[i]=d[i]
        for i in range(self.Max_iter):
            self.classes={}
            for i in range(self.k):
                self.classes[i]=[]
                
            for feature in d:
                classification=self.euclidean_distance(feature)
                self.classes[classification].append(feature)
            previous=dict(self.centroids)
            
            for cluster in self.classes:
                self.centroids[cluster]=np.average(self.classes[cluster],axis=0)
            
            optimal=True
            for centroid in self.centroids:
                original_centroid=previous[centroid]
                current_centroid=self.centroids[centroid]
                if np.sum((current_centroid-original_centroid)/original_centroid *100.0)>self.tolerance:
                    optimal=False
            if optimal:
                cluster=[]
                cluster_no=[]
                for k in self.classes:
                    l=len(self.classes[k])
                    for v in range(l):
                        cluster.append([self.classes[k][v],k])
                #print(cluster[2][1])
                clustered=pd.DataFrame(cluster,columns=["x","y"])
                break
        return clustered
    def euclidean_distance(self, data):
        distances = [np.sqrt(np.sum((data - self.centroids[centroid]))**2) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
    
    def manhattan_distance(self,data):
        distances = [np.sum(abs(data - self.centroids[centroid])) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
    def Minkowski_distance(self,data):
        distances = [np.sum(abs(data - self.centroids[centroid])**3)**(1/3) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
    def squared_euclidean_distance(self,data):
        distances = [np.sum((data - self.centroids[centroid]))**2 for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
def main():
    x, y = make_blobs(n_samples = 1000, centers = 3, n_features=2, shuffle=True, random_state=31)
    dataset=pd.DataFrame(x,columns=['x','y'])
    print(dataset.head())
    X = dataset.values 
    start = timeit.default_timer()
    #returns a numpy array
    km=K_Means(3,100)
    dataset=km.fit(X,dataset)
    stop=timeit.default_timer()
    print('Time: ', (stop - start)/60,'minutes')
    

	# Plotting starts here
    colors = {1:"red", 2:"green",3:"blue", 4:"cyan", 5:"magenta",6:"yellow"}

    for centroid in km.centroids:
        plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], s = 300,c=colors[centroid+1])

    for classification in km.classes:
        color = colors[classification+1]
        for features in km.classes[classification]:
            plt.scatter(features[0], features[1], color = color,s = 30)
    plt.show()
    
    
    # Using the elbow method to find the optimal number of clusters
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    
    #silhouette_score   

    sil = []
    kmax = 10

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2,kmax+1):
        kmeans = KMeans(n_clusters = k).fit(X)
        labels = kmeans.labels_
        sil.append(silhouette_score(x, labels, metric = 'euclidean'))
    plt.plot(range(2, 11), sil)
    plt.title('Silhouette Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('SIL')
    plt.show()
    
    return dataset

if __name__ == "__main__":
    dataset=main()