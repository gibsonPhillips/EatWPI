# Test Code
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Generate synthetic data
np.random.seed(42)  # For reproducibility
n_samples = 1000
n_features = 10

# Generate random features
X = np.random.randn(n_samples, n_features)

# Generate binary target labels (0 or 1)
# Let's create labels based on a linear combination of features plus some noise
weights = np.random.randn(n_features)
bias = np.random.randn(1)
y_prob = 1 / (1 + np.exp(-(np.dot(X, weights) + bias)))  # Sigmoid to get probabilities
y = (y_prob > 0.5).astype(int).flatten()  # Convert probabilities to binary labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Logistic Regression model
model = LogisticRegression(learning_rate=0.01, n_epochs=100, batch_size=32)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Output the results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)