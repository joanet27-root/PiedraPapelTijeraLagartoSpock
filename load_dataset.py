import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import pathlib

def load_dataset():

    # Ruta correcta a 'dataset'
    data_dir = pathlib.Path(
        r"..\dataset"
    )

    IMG_SIZE = (160, 160)
    BATCH_SIZE = 32
    SEED = 42

    # 1) Cargar datasets (train y val del MISMO sitio)
    train_ds = image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="training",
        seed=SEED
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="validation",
        seed=SEED
    )

    # class_names antes del prefetch
    class_names = train_ds.class_names
    print("Clases:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names, IMG_SIZE
