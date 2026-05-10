import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load Dataset
cancer = pd.read_csv(
    'https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv'
)

# Features & Target
y = cancer['diagnosis']
X = cancer.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.7,
    random_state=2529
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

# Train Model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Save Model & Scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model Saved Successfully")
