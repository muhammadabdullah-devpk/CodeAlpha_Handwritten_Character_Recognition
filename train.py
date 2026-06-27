from src.data_loader import load_mnist_data
from src.model_builder import build_cnn_model, train_model, save_model
from src.utils import ensure_dirs, plot_training_history, save_json, save_text_report
from src.config import REPORTS_DIR


def main():
    ensure_dirs()
    X_train, X_val, X_test, y_train, y_val, y_test = load_mnist_data()
    model = build_cnn_model(input_shape=(28, 28, 1), num_classes=10)

    history = train_model(model, X_train, y_train, X_val, y_val)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    save_model(model)

    plot_training_history(history, REPORTS_DIR / "training_history.png")
    save_json(
        REPORTS_DIR / "training_summary.json",
        {
            "model": "CNN",
            "test_loss": round(float(test_loss), 4),
            "test_accuracy": round(float(test_accuracy), 4),
            "epochs_completed": len(history.history["loss"]),
            "best_val_accuracy": round(float(max(history.history["val_accuracy"])), 4),
        },
    )
    save_text_report(
        REPORTS_DIR / "training_summary.txt",
        [
            "CodeAlpha - Handwritten Character Recognition",
            "Model: Convolutional Neural Network (CNN)",
            f"Test Accuracy: {test_accuracy:.4f}",
            f"Test Loss: {test_loss:.4f}",
            f"Epochs Completed: {len(history.history['loss'])}",
            f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}",
        ],
    )

    print("Training complete.")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Saved model to models/best_model.keras")


if __name__ == "__main__":
    main()
