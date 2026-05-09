from sklearn.preprocessing import MinMaxScaler, StandardScaler

# TODO: Remove unused func
def normalize(X):
    minmax = MinMaxScaler()
    zscore = StandardScaler()
    X_minmax = minmax.fit_transform(X)
    X_zscore = zscore.fit_transform(X)
    return X_minmax, X_zscore

def normalize_minmax(X):
    minmax = MinMaxScaler()
    X_minmax = minmax.fit_transform(X)
    return X_minmax

def normalize_zscore(X):
    zscore = StandardScaler()
    X_zscore = zscore.fit_transform(X)
    return X_zscore
