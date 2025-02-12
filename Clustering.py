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