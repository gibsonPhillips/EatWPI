import numpy as np

class SoftmaxRegression:
    """
    Softmax Regression classifier using Stochastic Gradient Descent (SGD)
    for multiclass classification problems.
    """

    def __init__(self, learning_rate=0.01, n_epochs=100, batch_size=32):
        """
        Initialize the Softmax Regression model.

        Parameters:
        -----------
        learning_rate : float
            The step size for updating model parameters
        n_epochs : int
            Number of passes through the training data
        batch_size : int
            Number of training examples to use in each gradient update
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.W = None  # weights
        self.b = None  # biases

    def softmax(self, z):
        """
        Compute the softmax function.

        Formula:
        softmax(z) = exp(z) / Σ exp(z)
        where z = Wx + b

        Parameters:
        -----------
        z : array-like of shape (n_samples, n_classes)
            Input values

        Returns:
        --------
        array-like of shape (n_samples, n_classes)
            Softmax probabilities for each class
        """
        # ===== Insert your code here =====
        z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))

        # Divide by exponentials for each row
        softmax_probs = z_exp / np.sum(z_exp, axis=1, keepdims=True)

        return softmax_probs

    def initialize_parameters(self, n_features, n_classes):
        """
        Initialize model parameters using Xavier initialization.

        Parameters:
        -----------
        n_features : int
            Number of input features
        n_classes : int
            Number of output classes
        """
        # Xavier initialization for weights
        # ===== Insert your code here =====

        limit = np.sqrt(2 / (n_features + n_classes))
        self.W = np.random.uniform(-limit, limit, (n_features, n_classes))

        # Initialize biases to zero
        self.b = np.zeros(n_classes)


    def compute_loss(self, y_true, y_pred):
        """
        Compute cross-entropy loss for multiclass classification.

        Formula:
        L = -1/N * Σ Σ(y * log(ŷ))
        where:
        - y is true label
        - ŷ is predicted probability
        - N is number of samples

        Parameters:
        -----------
        y_true : array-like of shape (n_samples, n_classes)
            True one-hot encoded labels
        y_pred : array-like of shape (n_samples, n_classes)
            Predicted probabilities

        Returns:
        --------
        float
            Average cross-entropy loss
        """
        # ===== Insert your code here =====
        # Clip y_pred to prevent log(0) errors and ensure numerical stability
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

        return loss



    def compute_gradients(self, X_batch, y_batch, y_pred):
        """
        Compute gradients for weights and biases.

        Formulas:
        ∂L/∂W = 1/N * X^T * (ŷ - y)
        ∂L/∂b = 1/N * Σ(ŷ - y)

        Parameters:
        -----------
        X_batch : array-like
            Input features for current batch
        y_batch : array-like
            True labels for current batch
        y_pred : array-like
            Predicted probabilities for current batch

        Returns:
        --------
        tuple
            Gradients for weights and biases
        """
        # ===== Insert your code here =====

        # current samples
        batch_size = X_batch.shape[0]

        # Calculate the error (difference between predicted and true labels)
        error = y_pred - y_batch

        # Gradient with respect to weights
        dW = (1 / batch_size) * np.dot(X_batch.T, error)

        # Gradient with respect to biases
        db = (1 / batch_size) * np.sum(error, axis=0)

        return dW, db


    def fit(self, X, y):
        """
        Train the softmax regression model using mini-batch SGD.

        Process:
        1. Initialize parameters
        2. For each epoch:
            a. Shuffle data
            b. Split into mini-batches
            c. For each mini-batch:
                - Compute forward pass (softmax)
                - Compute gradients
                - Update parameters

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values (class labels)
        """
        # Initialize Parameters
        # ===== Insert your code here =====
        n_samples,n_features = X.shape
        n_classes = len(np.unique(y))

        self.initialize_parameters(n_features, n_classes)

        y_one_hot = np.eye(n_classes)[y]

        # Epoch Loop
        for epoch in range(self.n_epochs):
            # Shuffle the data
            indices = np.arrange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y_one_hot[indices]

            # Mini-batch training
            for i in range(0, n_samples, self.batch_size):
                # Get Batch Data
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]


                # Forward pass
                z = np.dot(X_batch, self.W) + self.b
                y_pred = self.softmax(z)


                # Compute gradients
                dw,db = self.compute_gradients(X_batch, y_batch, y_pred)


                # Update parameters
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

            # Calculate and trace the loss
            z_all = np.dot(X, self.W) + self.b
            y_pred_all = self.softmax(z_all)
            loss = self.compute_loss(y_one_hot, y_pred_all)
            print(f"Epoch {epoch+1}/{self.n_epochs}, lose: {loss:.4f}")


    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples

        Returns:
        --------
        array-like of shape (n_samples, n_classes)
            Predicted probabilities for each class
        """
        # ===== Insert your code here =====
        z = np.dot(X, self.W) + self.b

        y_pred_proba = self.softmax(z)

        return y_pred_proba

    def predict(self, X):
        """
        Predict class labels for input samples.

        Formula:
        y = argmax(P(y|x))

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples

        Returns:
        --------
        array-like of shape (n_samples,)
            Predicted class labels
        """
        # ===== Insert your code here =====
        y_pred_proda = self.predict_proba(X)

        y_pred = np.argmax(y_pred_proda, axis=1)

        return y_pred