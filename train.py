# train.py
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils import plot_history   
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_model(model, train_ds, val_ds):
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[early_stop, lr_scheduler],
    )

    plot_history(history)
    model.save("modelo_final.keras")
    return history
