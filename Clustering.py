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

dataset.isnull().sum()

dataset.duplicated().sum()

numimputer = SimpleImputer(missing_values=np.nan, strategy='mean')
numimputer = numimputer.fit(dataset[[
    'DepTime','ArrTime','ActualElapsedTime', 'AirTime','ArrDelay','DepDelay', 'TaxiIn', 'TaxiOut'
    ]])
dataset[[
    'DepTime','ArrTime','ActualElapsedTime', 'AirTime','ArrDelay','DepDelay', 'TaxiIn', 'TaxiOut'
    ]] = numimputer.transform(dataset[[
        'DepTime','ArrTime','ActualElapsedTime', 'AirTime','ArrDelay','DepDelay', 'TaxiIn', 'TaxiOut'
        ]])
catimputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
catimputer = catimputer.fit(dataset[['TailNum']])
dataset[['TailNum']] = catimputer.transform(dataset[['TailNum']])

dataset.drop(['CancellationCode'], axis=1, inplace=True)

dataset.isnull().any().sum()

dataset.describe()

plt.figure(figsize=(10,5))
sns.histplot(x='DepTime', data=dataset, bins=30, color='royalblue', edgecolor='black', alpha=0.7)
plt.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='gray')
plt.xlabel('Departure Time (DepTime)', fontsize=16, fontweight='bold', color='darkblue')
plt.ylabel('Count of Flights', fontsize=16, fontweight='bold', color='darkblue')
plt.title('Distribution of Departure Times', fontsize=18, fontweight='bold', color='darkblue', pad=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(x='ArrTime', data=dataset, bins=30, color='maroon', edgecolor='black', alpha=0.7)
plt.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='gray')
plt.xlabel('Arrival Time (Arrtime)', fontsize=16, fontweight='bold', color='darkred')
plt.ylabel('Count of Flights', fontsize=16, fontweight='bold', color='darkred')
plt.title('Distribution of Arrival Times', fontsize=18, fontweight='bold', color='darkred', pad=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

plt.figure(figsize=(16, 6))

# Create a subplot for ArrDelay
plt.subplot(1, 2, 1)
sns.boxplot(x='ArrDelay', data=dataset, color='skyblue')
plt.title('Box Plot of Arrival Delay (ArrDelay)', fontsize=14, fontweight='bold')
plt.xlabel('Arrival Delay (minutes)', fontsize=12)
plt.xticks(rotation=45)

# Create a subplot for DepDelay
plt.subplot(1, 2, 2)
sns.boxplot(x='DepDelay', data=dataset, color='salmon')
plt.title('Box Plot of Departure Delay (DepDelay)', fontsize=14, fontweight='bold')
plt.xlabel('Departure Delay (minutes)', fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

# Create the scatter plot
sns.scatterplot(x='DepDelay', y='ArrDelay', data=dataset, alpha=0.5, color='royalblue', edgecolor='black')

# Add labels and title
plt.xlabel('Departure Delay (minutes)', fontsize=14, fontweight='bold')
plt.ylabel('Arrival Delay (minutes)', fontsize=14, fontweight='bold')
plt.title('Scatter Plot of Departure Delay vs Arrival Delay', fontsize=16, fontweight='bold')

# Add a grid for better readability
plt.grid(visible=True, linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# Calculate the correlation matrix
correlation_matrix = dataset[['DepDelay', 'ArrDelay', 'TaxiIn', 'TaxiOut', 'ActualElapsedTime', 'AirTime', 'Distance']].corr()

# Set up the figure size
plt.figure(figsize=(10, 8))

# Create the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=0.5)

# Add title
plt.title('Correlation Heatmap of Flight Features', fontsize=16, fontweight='bold')

# Show the plot
plt.show()

features = ['DepDelay', 'ArrDelay', 'TaxiIn', 'TaxiOut', 'ActualElapsedTime', 'AirTime', 'Distance']

# Creating the pair plot
plt.figure(figsize=(12, 12))
sns.pairplot(dataset[features], diag_kind='kde', plot_kws={'alpha':0.5, 's':20, 'edgecolor':'k'}, markers="o")

# Show the plot
plt.suptitle('Pair Plot of Flight Features', y=1.02, fontsize=16, fontweight='bold')
plt.show()

# Define thresholds for outliers
arr_delay_threshold = 300
dep_delay_threshold = 300

# Filter out outliers in ArrDelay and DepDelay
dataset = dataset[(dataset['ArrDelay'] <= arr_delay_threshold) &
                          (dataset['DepDelay'] <= dep_delay_threshold)]

dataset['Total Delay'] = dataset['ArrDelay'] + dataset['DepDelay']

dataset['Taxi Time'] = dataset['TaxiIn'] + dataset['TaxiOut']

encoder = preprocessing.OrdinalEncoder()
dataset['UniqueCarrier'] = encoder.fit_transform(dataset['UniqueCarrier'].values.reshape(-1,1))

dataset['TailNum'] = encoder.fit_transform(dataset['TailNum'].values.reshape(-1,1))

dataset['Origin'] = encoder.fit_transform(dataset['Origin'].values.reshape(-1,1))

dataset['Dest'] = encoder.fit_transform(dataset['Dest'].values.reshape(-1,1))

X = dataset.iloc[:,[4,5,6,7,10,11,14,15,16,21,22]].values
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init= 'k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Elbow Method for Optimal k')
plt.show()

kmeans = KMeans(n_clusters = 2, init= 'k-means++', random_state= 0)

# Test Silhouette Score for different numbers of clusters
for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f'Silhouette Score for n_clusters={n_clusters}: {silhouette_avg}')
    pca = PCA(n_components=3)

X_reduced = pca.fit_transform(X)

pca.explained_variance_ratio_

sum(pca.explained_variance_ratio_)

variance = np.round(pca.explained_variance_ratio_*100, decimals = 1)

plt.figure(figsize = (10,6))
plt.plot(range(1, len(variance)+1), variance.cumsum(), marker = "o", linestyle = "--")
plt.grid()
plt.ylabel("% Variance Sum")
plt.xlabel("Components")
plt.title("Variance by Component")
plt.show()

y_kmeans = kmeans.fit_predict(X_reduced)

# Plotting the PCA-transformed data in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for the data points
sc = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
                c=y_kmeans, cmap='rainbow', s=50, edgecolor='k')

# Scatter plot for the cluster centroids
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           kmeans.cluster_centers_[:, 2], s=100, c='Black', label='Centroids', marker='x')

# Labels and title
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D K-Means Clustering on PCA-Reduced Data')
ax.legend()

# Show plot with color bar
plt.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
plt.show()

neighbours = NearestNeighbors(n_neighbors=2)
distances, indices = neighbours.fit(X).kneighbors(X)

distances = distances[:,1]
distances = np.sort(distances, axis=0)

plt.plot(distances)
plt.show()

dbscan = DBSCAN(eps=0.75, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

# Plotting the PCA-transformed data in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for the data points with DBSCAN clustering
sc = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y_dbscan, cmap='rainbow', s=50, edgecolor='k')

# Labels and title
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('DBSCAN Clustering on PCA-Reduced Data (3D)')

# Show plot with color bar
plt.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
plt.show()