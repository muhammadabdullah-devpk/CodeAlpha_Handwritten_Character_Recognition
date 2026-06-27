import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from src.data_loader import load_mnist_data
from src.model_builder import load_saved_model
from src.utils import ensure_dirs, plot_confusion_matrix, save_json, save_text_report
from src.config import REPORTS_DIR


def main():
    ensure_dirs()
    _, _, X_test, _, _, y_test = load_mnist_data()
    model = load_saved_model()

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    plot_confusion_matrix(cm, REPORTS_DIR / "confusion_matrix.png")
    save_json(
        REPORTS_DIR / "metrics.json",
        {
            "accuracy": round(float(accuracy), 4),
            "precision_weighted": round(float(precision), 4),
            "recall_weighted": round(float(recall), 4),
            "f1_weighted": round(float(f1), 4),
            "roc_auc_ovr": round(float(roc_auc), 4),
        },
    )
    save_text_report(
        REPORTS_DIR / "classification_report.txt",
        [
            f"Accuracy: {accuracy:.4f}",
            f"Precision (weighted): {precision:.4f}",
            f"Recall (weighted): {recall:.4f}",
            f"F1-Score (weighted): {f1:.4f}",
            f"ROC-AUC (OvR): {roc_auc:.4f}",
            "",
            "Detailed Classification Report:",
            report,
        ],
    )

    print("Evaluation complete.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()
