import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

# Load the data
data = pd.read_csv("train.csv")

# Drop some high-missing columns
data = data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

# Handle missing values
data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].median())
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
data = data.dropna()

# One-hot encoding
data = pd.get_dummies(data)

# Split features and target
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
preds = model.predict(X_test)

# Option 1 fix: Manually calculate RMSE
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)

print("Random Forest RMSE:", rmse)

# Save the model
joblib.dump(model, 'house_price_model.pkl')
print("Model saved successfully as house_price_model.pkl")
