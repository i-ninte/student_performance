import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import gzip

# Load the dataset
df = pd.read_csv('performance.csv')

# Drop rows with NaN values
df.dropna(inplace=True)

# Convert categorical data
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'yes': 1, 'no': 0})

# Features and target
X = df[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
y = df['Performance Index']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the model (compressed)
with gzip.open('rf_model.pkl.gz', 'wb') as f:
    joblib.dump(model, f)
