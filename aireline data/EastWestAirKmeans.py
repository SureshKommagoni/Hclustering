import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

EastWestAir = pd.read_csv("file:///D:/ExcelR/Assignments/HClustering/EastWestAirlines.csv")

# normalization function
def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return()

# Normalized data frame (considering part of the data)

df_norm = norm_func(EastWestAir.iloc[:,1:])

## scree plot or elbow curve

k = list(range(2,8))
k

TWSS =[] # variable for storing total within sum of squares for each kmeans

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each k means
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]), "euclidean")))
    TWSS.append(sum(WSS))

        









