# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 22:51:34 2017

@author: heyup
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import NMF

# Import data
df = pd.read_csv('artsy_project_update.csv', index_col=1, header=0)
df = df.iloc[:, 1:15]
df.describe()
df.info()
df.head()

# Create a MaxAbsScaler
scaler = MaxAbsScaler()
# Create an NMF model and dictate 5 components
nmf = NMF(n_components=5)
# Create a Normalizer
normalizer = Normalizer()
# Create a pipeline
pipeline = make_pipeline(scaler, nmf, normalizer)
norm_features = pipeline.fit_transform(df.iloc[:, 0:13])
# Interpret NMF components
df_components = pd.DataFrame(nmf.components_, columns=df.iloc[:, 0:13].columns.values)
component1 = df_components.loc[0, :].nlargest()
#
component2 = df_components.loc[1, :].nlargest()
#
component3 = df_components.loc[2, :].nlargest()
#
component4 = df_components.loc[3, :].nlargest()
#
component5 = df_components.loc[4, :].nlargest()
#


# Build a recommender system which recommends similar customers based on NMF.

def similarCustomers(customerID):
   "returns similar customers based on NMF components"
   # Create a DataFrame
   df_nmf = pd.DataFrame(norm_features, index=df.index.values)
   # Select row of customer 
   customer = df.loc[customerID,:].iloc[0:13]
   # Compute cosine similarities
   similarities = df.iloc[:, 0:13].dot(customer)
   # Display those with highest cosine similarity
   print(similarities.nlargest())
   print(df_nmf.loc[customerID])

