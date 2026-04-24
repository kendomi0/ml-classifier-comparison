import numpy as np
from preprocessing import normalize_minmax, normalize_zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import datasets

def test_normalize_minmax():
    X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8)
    result = normalize_minmax(X)
    assert np.array_equal(result, MinMaxScaler().fit_transform(X))

def test_normalize_zscore():
    X, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=42, factor=0.8)
    result = normalize_zscore(X)
    assert np.array_equal(result, StandardScaler().fit_transform(X))