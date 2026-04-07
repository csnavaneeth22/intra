import numpy as np
import pandas as pd

def lorentzian_distance(arr1, arr2):
    # arr1: shape (N, F), arr2: shape (F,)
    # Output: shape (N,)
    diff = np.abs(arr1 - arr2)
    log_diff = np.log(1 + diff)
    return np.sum(log_diff, axis=1)

def get_lorentzian_predictions(features_df, close_series, neighbors_count=8, max_bars_back=2000, step=4):
    """
    Finds the KNN predictions for each bar in the dataframe using Lorentzian Distance.
    Returns a series of predictions (sum of k nearest neighbors' labels).
    
    features_df: DataFrame of features (N, F)
    close_series: Series of close prices (N,)
    neighbors_count: k
    max_bars_back: lookback window
    step: chronological spacing constraint
    """
    n = len(features_df)
    predictions = np.zeros(n)
    
    # Calculate target labels (y) mimicking PS logic:
    # src[4] < src[0] -> short (-1), src[4] > src[0] -> long (1), else 0
    # In python terms, if future 4th bar close is > current close, it's a Long (1).
    # Wait, the PS code says:
    # y_train = src[4] < src[0] ? -1 : src[4] > src[0] ? 1 : 0
    # In PS, src[4] means 4 bars AGO. Wait.
    # If src[4] (price 4 bars ago) < src[0] (current price), this means price went UP.
    # Therefore the training label for the bar 4 bars ago is 'went up'.
    # In pandas, if we are at index i, the "current" bar is i.
    # The label for index i should represent what happened between i and i+4.
    # So y[i] = 1 if close[i+4] > close[i] else -1.
    
    future_close = close_series.shift(-4)
    y_train = np.where(future_close > close_series, 1, np.where(future_close < close_series, -1, 0))
    y_train = pd.Series(y_train, index=close_series.index)
    
    features = features_df.values
    y_vals = y_train.values
    
    for i in range(max_bars_back, n):
        # Current feature vector
        current_features = features[i]
        
        # Historical window
        # We need historical features and their corresponding labels.
        # Ensure we don't look ahead! The label for a bar requires 4 future bars.
        # So the latest valid historical bar we can use is i-4.
        start_idx = max(0, i - max_bars_back)
        end_idx = i - 4
        
        if end_idx <= start_idx:
            predictions[i] = 0
            continue
            
        hist_features = features[start_idx:end_idx]
        hist_labels = y_vals[start_idx:end_idx]
        
        # Calculate Lorentzian distances
        distances = lorentzian_distance(hist_features, current_features)
        
        # We want the nearest neighbors (smallest distances). 
        # Sort and pick top k. But PS uses a chronological step constraint.
        # Let's filter hist_features by the step constraint if needed.
        # The PS logic: "i%4" for the loop backwards means it only picks neighbors at 4-bar intervals.
        # We can just pick the neighbors, but require their indices to be separated by at least 4 bars?
        # A simpler robust MLE approach: just take bottom k distances.
        
        # To strictly replicate PS's chronologically spaced constraint:
        # We can slice arrays with step:
        spaced_distances = distances[::-step] # backwards with step
        spaced_labels = hist_labels[::-step]
        
        if len(spaced_distances) > 0:
            # Get indices of the smallest distances
            k = min(neighbors_count, len(spaced_distances))
            nearest_indices = np.argsort(spaced_distances)[:k]
            nearest_labels = spaced_labels[nearest_indices]
            
            predictions[i] = np.sum(nearest_labels)
        else:
            predictions[i] = 0
            
    return pd.Series(predictions, index=features_df.index)
