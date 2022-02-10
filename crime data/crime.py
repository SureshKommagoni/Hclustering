import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

crime = pd.read_csv("file:///D:/DATA SCIENCE/ExcelR/Assignments/HClustering/crime_data.csv")

# normalization function

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)\

# normalized data (considering the numerical part of the data)
    
df_norm = norm_func(crime.iloc[:,1:])

from scipy.cluster.hierarchy import linkage

import scipy.cluster.hierarchy as sch # for creating dendrogram

type(df_norm)

z = linkage(df_norm, method="complete", metric = "euclidean")
type(z)

plt.figure(figsize=(15,5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')

sch.dendrogram(
        z,
        leaf_rotation =0., # rotates the x axis lables
        leaf_font_size=8.,# font size for the x axis lales
)
plt.show()

# applying agglomerative clustering choosing 4 clusters from the dendrogram

from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering( n_clusters = 4, linkage = "complete", affinity = "euclidean").fit(df_norm)
h_complete.labels_

cluster_lables =pd.Series(h_complete.labels_)

crime['clust'] = cluster_lables # creating a new column and assigning it to new column
crime = crime.iloc[:,[5,0,1,2,3,4]]

# getting aggregate mean for the each cluster

crime.iloc[:,2:].groupby(crime.clust).mean()

# creating csv file

crime.to_csv("crimedata.csv", encoding="utf-8")


import os
os.getcwd()
os.chdir("D:\\CODES")










    