from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize(X):
    minmax = MinMaxScaler()
    zscore = StandardScaler()
    X_minmax = minmax.fit_transform(X)
    X_zscore = zscore.fit_transform(X)
    return X_minmax, X_zscore