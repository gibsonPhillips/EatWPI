#------------ No New Package --------------
import numpy as np
#---------------------------------------------------------

class LogisticRegression:
    """
    Logistic Regression classifier using Stochastic Gradient Descent (SGD)
    for binary classification problems.
    """

    def __init__(self, learning_rate=0.01, n_epochs=100, batch_size=32):
        """
        Initialize the Logistic Regression model.

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
        self.w = None  # weights
        self.b = None  # bias

    def sigmoid(self, z):
        """
        Compute the sigmoid activation function.

        Formula:
        σ(z) = 1 / (1 + e^(-z))
        where z = wx + b

        Parameters:
        -----------
        z : array-like
            Input values

        Returns:
        --------
        array-like
            Sigmoid of input values
        """
        # ===== Insert your code here =====
        result = []
        for Zed in z:
            result.append(1 / (1 + np.exp(-Zed)))
        return []

    def initialize_parameters(self, n_features):
        """
        Initialize model parameters using Xavier initialization.

        Formula for Xavier initialization:
        w ~ N(0, sqrt(2/n_features))

        Parameters:
        -----------
        n_features : int
            Number of input features
        """
        # Xavier initialization for better convergence
        # ===== Insert your code here =====


    def compute_loss(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss.

        Formula:
        L = -1/N * Σ(y * log(ŷ) + (1-y) * log(1-ŷ))
        where:
        - y is true label
        - ŷ is predicted probability
        - N is number of samples

        Parameters:
        -----------
        y_true : array-like
            True binary labels
        y_pred : array-like
            Predicted probabilities

        Returns:
        --------
        float
            Average binary cross-entropy loss
        """
        # ===== Insert your code here =====

    def compute_gradients(self, X_batch, y_batch, y_pred):
        """
        Compute gradients for weights and bias.

        Formulas:
        ∂L/∂w = 1/N * X^T * (ŷ - y)
        ∂L/∂b = 1/N * Σ(ŷ - y)
        where:
        - X is input features
        - y is true labels
        - ŷ is predicted probabilities
        - N is batch size

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
            Gradients for weights and bias
        """
        # ===== Insert your code here =====

    def fit(self, X, y):
        """
        Train the logistic regression model using mini-batch SGD.

        Process:
        1. Initialize parameters
        2. For each epoch:
            a. Shuffle data
            b. Split into mini-batches
            c. For each mini-batch:
                - Compute forward pass (sigmoid)
                - Compute gradients
                - Update parameters

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
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

        Formula:
        P(y=1|x) = σ(wx + b)
        where σ is the sigmoid function

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples

        Returns:
        --------
        array-like of shape (n_samples,)
            Predicted probabilities
        """
        # ===== Insert your code here =====

    def predict(self, X, threshold=0.5):
        """
        Predict class labels for input samples.

        Formula:
        y = 1 if P(y=1|x) >= threshold else 0

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
        threshold : float
            Classification threshold

        Returns:
        --------
        array-like of shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        # ===== Insert your code here =====