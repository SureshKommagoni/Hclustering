import pandas as pd
import numpy as np
import matplotlib.pylab as plt

EastWest = pd.read_csv("file:///D:/ExcelR/Assignments/HClustering/EastWestAirlines.csv")

# normalized function

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

# normalized data frame (considering the numerical part of the data)

df_norm = norm_func(EastWest.iloc[:,1:])

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch # for creating dendrogram

type(df_norm)

z = linkage(df_norm, method = "complete", metric = "euclidean")
type(z)

plt.figure(figsize=(15,5)); plt.title("Hierarchichal Clustering Dendrogram"); plt.xlabel("Index");plt.ylabel("Distance")

sch.dendrogram(
        z,
        leaf_rotation=0., # rotates the x axis lables
        leaf_font_size=8.,# font size for the x axis lables
)

plt.show()

# now applying AggloremativeClusting choosing 4 clusters from the above shown dendrogram

from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters=5, linkage="complete", affinity = "euclidean").fit(df_norm)
h_complete.labels_

cluster_lables = pd.Series(h_complete.labels_)
EastWest['clust'] = cluster_lables
EastWest = EastWest.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]

# getting aggregate mean of each cluster

EastWest.iloc[:,2:].groupby(EastWest.clust).mean()

# creating csv file
EastWest.to_csv("EastWestnew.csv", encoding = "utf-8") # utf = universal text format

import os
os.getcwd()
os.chdir("D://CODES")















