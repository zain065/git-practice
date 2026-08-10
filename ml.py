import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dataset
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8])
marks = np.array([45, 50, 55, 65, 70, 75, 85, 90])

# Features and target
X = hours.reshape(-1, 1)
y = marks

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

print("Actual:", y_test)
print("Predicted:", predictions)

# Evaluate
score = r2_score(y_test, predictions)

print("R² Score:", score)

# Predict new value
new_prediction = model.predict([[10]])

print("Predicted marks for 10 hours:", new_prediction)
print("Slope (Coefficient):", model.coef_)
print("Intercept:", model.intercept_)
