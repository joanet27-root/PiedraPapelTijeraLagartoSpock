# model_cnn.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    InputLayer,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    Rescaling,
)
from tensorflow.keras.optimizers import Adam


def build_model(num_classes, IMG_SIZE):

    model = Sequential()

    model.add(InputLayer(input_shape=IMG_SIZE + (3,)))

    # Aumentación
    model.add(RandomFlip("horizontal"))
    model.add(RandomRotation(0.1))
    model.add(RandomZoom(0.1))

    # Normalización
    model.add(Rescaling(1.0 / 255.0))

    # Bloque 1
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Bloque 2
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Bloque 3
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Clasificador
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=1e-3),  # subimos un poco el LR
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model
