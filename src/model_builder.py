from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from src.config import BEST_MODEL_PATH


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=1),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=128,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def save_model(model):
    model.save(BEST_MODEL_PATH)


def load_saved_model():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError("Saved model not found. Please run: python train.py")
    return load_model(BEST_MODEL_PATH)
