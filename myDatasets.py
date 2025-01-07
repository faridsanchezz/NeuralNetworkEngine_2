import numpy as np
from mnist import MNIST
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode(y):
    """
    One-hot encode a vector of labels using sklearn's OneHotEncoder.

    Parameters:
    -----------
    y : np.ndarray
        Array of labels.

    Returns:
    --------
    np.ndarray
        One-hot encoded labels.
    """
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse_output to avoid FutureWarning
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))  # Reshape to 2D array
    return y_encoded


def load_and_preprocess_data_iris():
    """
    Load and preprocess the Iris dataset.

    Returns:
    --------
    tuple
        X_train, y_train, X_test, y_test
    """
    iris = datasets.load_iris()
    X = iris.data  # Features
    y = iris.target  # Labels

    # Normalize features
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

    # One-hot encode labels
    y_encoded = one_hot_encode(y)

    # Split dataset using sklearn's train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test



def load_and_preprocess_data_mnist(path_mnist_data):
    """
    Load and preprocess the MNIST dataset.

    Parameters:
    -----------
    path_mnist_data : str
        Path to MNIST data.

    Returns:
    --------
    tuple
        X_train, y_train, X_test, y_test
    """
      # ruta de la carpeta donde se encuentra los datos MNIST
    try:
        mndata = MNIST(path_mnist_data)
        X_train, y_train = mndata.load_training()
        X_test, y_test = mndata.load_testing()

        # Normalize features
        X_train, X_test = np.array(X_train) / 255.0, np.array(X_test) / 255.0

        # Convert labels to numpy arrays and one-hot encode
        y_train = one_hot_encode(np.array(y_train))
        y_test = one_hot_encode(np.array(y_test))

        return X_train, X_test, y_train, y_test

    except Exception as e:
        raise FileNotFoundError(f"Error loading MNIST data. Check the path: {path_mnist_data}. Error: {str(e)}")




    
