import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Veri setini yükle
data = pd.read_csv("breast_cancer.csv")
data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})
data = data.drop(['id', 'Unnamed: 32'], axis=1)

# Seçilen özellikler
selected_features = ['concave points_worst', 'perimeter_worst', 'concave points_mean', 'radius_worst', 'perimeter_mean']
X = data[selected_features]
y = data['diagnosis']

# Veri setini ayır ve normalize et
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modeli eğit
model = LogisticRegression()
model.fit(X_train, y_train)

# Modeli ve scaler'ı kaydet
with open("model.pkl", "wb") as f:
    pickle.dump((scaler, model), f)
