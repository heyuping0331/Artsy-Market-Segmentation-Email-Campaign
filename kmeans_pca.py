# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Import data
df = pd.read_csv('artsy_project_update.csv', index_col=1, header=0)
df = df.iloc[:, 1:15]
df.describe()
df.info()
df.head()

# Quantitatively determine the number of clusters by inertia.
inertias = []

for k in range(1,10):
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k) 
    # Standardized features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.iloc[:, 0:13])
    # Fit model to samples
    model.fit(df_scaled)   
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot K vs inertias
plt.plot(range(1,10), inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(range(1,10))
plt.show()

# Fit a Kmeans clustering model with 6 clusters
np.random.seed(42)
scaler = StandardScaler()
kmeans = KMeans(n_clusters=6)
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(df.iloc[:, 0:13])
# Calculate the cluster labels: labels
labels = pipeline.predict(df.iloc[:, 0:13])
df['Cluster'] = labels
  
  
# Export to Excel
#df.to_excel("Kmeans Cluster.xlsx") 

# Visualize the clustering result via PCA
pca = PCA()
pipeline2 = make_pipeline(scaler, pca)
pipeline2.fit(df.iloc[:, 0:13])
# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show() # Now we know it won't lose much information to project data into 2D.
# Re-define PCA
pca = PCA(n_components=2)
pca.fit(df.iloc[:, 0:13])
pca_features = pca.transform(df.iloc[:, 0:13])
# Print the shape of pca_features
print(pca_features.shape)       
# Finally plot the PCA features to visualize clustering performance.
plt.scatter(pca_features[:,0], pca_features[:,1], c=labels)
plt.show()







