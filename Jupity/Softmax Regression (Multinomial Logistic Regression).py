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

        # Epoch Loop
        for epoch in range(self.n_epochs):
            # Shuffle the data
            # ===== Insert your code here =====

            # Mini-batch training
            for i in range(0, n_samples, self.batch_size):
                # Get Batch Data
                # ===== Insert your code here =====

                # Forward pass
                # ===== Insert your code here =====

                # Compute gradients
                # ===== Insert your code here =====

                # Update parameters
                # ===== Insert your code here =====

            # Calculate and trace the loss
            # ===== Insert your code here =====


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
