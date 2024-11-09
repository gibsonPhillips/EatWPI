import numpy as np
import requests
import gzip
import os
from urllib.request import urlretrieve
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# URLs to download the Fashion MNIST dataset
BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FILENAMES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# Download and extract Fashion MNIST dataset
def download_and_load_mnist(filename, num_items, item_size, reshape_dims=None):
    if not os.path.exists(filename):
        urlretrieve(BASE_URL + filename.split("/")[-1], filename)

    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=item_size)
        if reshape_dims:
            return data.reshape(num_items, *reshape_dims)
        else:
            return data

# Load datasets
def load_fashion_mnist():
    train_images = download_and_load_mnist("train-images-idx3-ubyte.gz", 60000, 16, (28, 28))
    train_labels = download_and_load_mnist("train-labels-idx1-ubyte.gz", 60000, 8)
    test_images = download_and_load_mnist("t10k-images-idx3-ubyte.gz", 10000, 16, (28, 28))
    test_labels = download_and_load_mnist("t10k-labels-idx1-ubyte.gz", 10000, 8)

    # Reshape and normalize images
    train_images = train_images.reshape(60000, 28*28) / 255.0
    test_images = test_images.reshape(10000, 28*28) / 255.0

    return train_images, train_labels, test_images, test_labels

# One-hot encoding for labels
def one_hot_encode(labels, num_classes=10):
    encoder = OneHotEncoder(categories=[range(num_classes)], sparse=False)
    return encoder.fit_transform(labels.reshape(-1, 1))

# Load data
X_train, y_train, X_test, y_test = load_fashion_mnist()
y_train_encoded = one_hot_encode(y_train)
y_test_encoded = one_hot_encode(y_test)

# Initialize and train the Softmax Regression model
model = SoftmaxRegression(learning_rate=0.1, n_epochs=50, batch_size=64)
model.fit(X_train, y_train_encoded)

# Predict on the test set
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
