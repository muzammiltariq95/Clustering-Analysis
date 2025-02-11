# !pip install numpy pandas seaborn matplotlib scikit-learn 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN


# %matplotlib inline

import os
os.environ['OMP_NUM_THREADS'] = '1'

dataset = pd.read_csv('hflights.csv')

dataset.shape

dataset.head()

dataset.info()