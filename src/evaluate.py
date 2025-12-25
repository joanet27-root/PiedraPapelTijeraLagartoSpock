import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from utils import plot_confusion_matrix

def evaluate_model(model, val_ds, class_names):

    # Evaluación global
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Loss validación: {val_loss:.4f}")
    print(f"Accuracy validación: {val_acc:.4f}")

    # Predicciones
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriz de confusión:")
    print(cm)

    print("\nInforme de clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion_matrix(cm, class_names)
