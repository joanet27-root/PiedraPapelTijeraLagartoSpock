# main.py
from load_dataset import load_dataset
from model_cnn import build_model
from train import train_model
from evaluate import evaluate_model
import json  


def main():
    print("Cargando dataset...")
    train_ds, val_ds, class_names, IMG_SIZE = load_dataset()

    # Guardamos el orden de las clases
    with open("class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f)
    print("Orden de clases guardado en class_names.json:", class_names)

    print("Creando modelo...")
    model = build_model(len(class_names), IMG_SIZE)

    print("Entrenando modelo...")
    history = train_model(model, train_ds, val_ds)

    print("Evaluando modelo...")
    evaluate_model(model, val_ds, class_names)

if __name__ == "__main__":
    main()
