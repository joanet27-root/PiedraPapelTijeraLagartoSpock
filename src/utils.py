import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def plot_history(history):
    """
    Dibuja las curvas de entrenamiento y validación
    para loss y accuracy a partir del objeto history de Keras.
    """
    # --- Accuracy ---
    acc = history.history.get("accuracy")
    val_acc = history.history.get("val_accuracy")

    # --- Loss ---
    loss = history.history.get("loss")
    val_loss = history.history.get("val_loss")

    epochs = range(1, len(acc) + 1)

    # Accuracy
    plt.figure()
    plt.plot(epochs, acc, label="Entrenamiento")
    plt.plot(epochs, val_acc, label="Validación")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title("Accuracy de entrenamiento y validación")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Loss
    plt.figure()
    plt.plot(epochs, loss, label="Entrenamiento")
    plt.plot(epochs, val_loss, label="Validación")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Loss de entrenamiento y validación")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names):
    """
    Dibuja la matriz de confusión a partir de cm
    (numpy array) y la lista de nombres de clases.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Matriz de confusión")
    plt.tight_layout()
    plt.show()
