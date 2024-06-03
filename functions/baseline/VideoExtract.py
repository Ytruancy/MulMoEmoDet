import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import subprocess
import os
from sklearn.preprocessing import LabelEncoder


def histogram_features(movements: np.ndarray, num_bins: int = 10) -> np.ndarray:
    """
    Transform a list of normalized movement volumes into histogram bins. Each movement is 
    represented as a count of values falling into each bin range.

    Parameters:
    - movements (np.ndarray): A 2D array where each row is a normalized movement sequence.
    - num_bins (int): Number of bins to divide the range [0, 1] into.

    Returns:
    - np.ndarray: A 2D array where each row represents the histogram bin counts for a movement.
    """
    # Define the bin edges for the histogram
    bins = np.linspace(0, 1, num_bins + 1)
    
    # Initialize an array to store histogram counts for each movement
    histogram_counts = []
    
    for movement in movements:
        # Calculate the histogram for each movement
        count, _ = np.histogram(movement, bins=bins)
        histogram_counts.append(count)
    
    return np.array(histogram_counts)


def calculate_movement(features):
    distances = []
    for i in range(68):
        x = features[f' X_{i}']
        y = features[f' Y_{i}']
        z = features[f' Z_{i}']
        distance = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
        distances.append(distance)
    distances = np.array(distances)
    scaler = MinMaxScaler()
    distances = scaler.fit_transform(distances.T).T
    return distances

def polynomial_regression(movements):
    from numpy.polynomial import Polynomial
    polynomials = []
    for movement in movements:
        p = Polynomial.fit(range(len(movement)), movement, 25)
        polynomials.append(p.convert().coef)
    polynomials = np.array(polynomials)
    return polynomials


def video_extract(video_feature_path):
    """
    Provide the extracted openface csv path, return builded video feature set
    """
    #Extract baseline video features
    video_features = pd.read_csv(video_feature_path)
    landmarks = video_features[[f" X_{i}" for i in range(68)] +
                                   [f" Y_{i}" for i in range(68)] +
                                   [f" Z_{i}" for i in range(68)]]
    movements = calculate_movement(landmarks)
    poly_features = histogram_features(movements)
    poly_features = torch.tensor(poly_features, dtype=torch.float32)
    
    return poly_features
