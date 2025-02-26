from sklearn.preprocessing import LabelEncoder

def label_encode(df):
    le = LabelEncoder()
    return df.apply(le.fit_transform)
