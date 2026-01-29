import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
import os

def prepare_data_for_segmentation(df):
    """
    Prepare data for K-Means algorithm.
    
    Select value and satisfaction columns, normalize the data 
    and return the scaled data and the used columns.
    """
    # Selecting columns with value and satisfaction metrics
    columns = ['Total', 'Rating', 'gross income']
    data = df[columns]
    
    # Normalizing the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, columns

def find_optimal_clusters(scaled_data):
    """
    Uses the Elbow method with KneeLocator to find the optimal number of clusters (K).
    
    The Elbow method is a heuristic for determining K in K-Means clustering.
    It computes the sum of squared errors for each K from 1 to n_clusters
    and then plots the points on a graph. The location of a "knee" in the graph
    is the optimal number of clusters.
    """
    # Calculate the sum of squared errors for each K from 1 to 11
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        sse.append(kmeans.inertia_)
    
    # Use KneeLocator to find the optimal K
    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    return kl.elbow

def apply_kmeans(df, scaled_data, n_clusters):
    """
    Applies the K-Means algorithm to the scaled data and assigns the clusters to the original dataframe.

    Parameters:
    df (pandas.DataFrame): The original dataframe.
    scaled_data (numpy.ndarray): The scaled data.
    n_clusters (int): The number of clusters to find.

    Returns:
    tuple: A tuple containing the dataframe with the clusters assigned and the KMeans object.
    """
    # Create a KMeans object with the specified number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    # Fit the data and predict the clusters
    df['Cluster'] = kmeans.fit_predict(scaled_data)
    
    # Return the dataframe with the clusters assigned and the KMeans object
    return df, kmeans
