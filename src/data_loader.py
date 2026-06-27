from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def load_mnist_data(random_state=42):
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    X_train_full = X_train_full.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    X_train_full = X_train_full[..., None]
    X_test = X_test[..., None]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.1,
        stratify=y_train_full,
        random_state=random_state,
    )

    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)

    return X_train, X_val, X_test, y_train, y_val, y_test
