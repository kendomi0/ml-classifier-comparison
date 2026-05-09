from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize_minmax(X):
    minmax = MinMaxScaler()
    X_minmax = minmax.fit_transform(X)
    return X_minmax

def normalize_zscore(X):
    zscore = StandardScaler()
    X_zscore = zscore.fit_transform(X)
    return X_zscore
