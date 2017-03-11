
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE


# Import data
df = pd.read_csv('artsy_project_update.csv', index_col=1, header=0)
df = df.iloc[:, 1:15]
df.describe()
df.info()
df.head()


# Normalize 
df_normalized = normalize(df.iloc[:, 0:13])
# Hierarchical clusterig iwth COMPLETE linkage
mergings = linkage(df_normalized, method='complete')
# Plot the dendrogram
dendrogram(mergings, leaf_rotation=90, leaf_font_size=6)
plt.show()
# Cut the dendrogram to create 7 cluster and generate cluster labels
labels = fcluster(mergings, 7, criterion='maxclust')
# Export to Excel
# df['Cluster'] = labels
# df.to_excel('Hierarchical Cluster.xlsx')

# t-TSNE model
tsne = TSNE(learning_rate=300)
tsne_features = tsne.fit_transform(df_normalized)
# Select coordinates
xs = tsne_features[:,0]
ys = tsne_features[:,1]
# Scatter plot, coloring by cluster labels
plt.scatter(xs, ys, c=labels)
plt.show()
# It seems the clusters are not quite differentiated from each other in 2D.

# NMF model
nmf = NMF(n_components=2)
nmf_features = nmf.fit_transform(df_normalized)
# Select coordinates
xs = nmf_features[:,0]
ys = nmf_features[:,1]
# Scatter plot, coloring by cluster labels
plt.scatter(xs, ys, c=labels)
plt.show()






