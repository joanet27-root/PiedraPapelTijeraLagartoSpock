import os
from PIL import Image

base = "../dataset"

print("\n CONTANDO IMÁGENES POR CLASE:")
for cls in sorted(os.listdir(base)):
    path = os.path.join(base, cls)
    if not os.path.isdir(path): continue

    files = [f for f in os.listdir(path)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"{cls:10s} → {len(files)} imágenes")

print("\n COMPROBANDO QUE TODAS LAS IMÁGENES ABREN...")
for cls in sorted(os.listdir(base)):
    path = os.path.join(base, cls)
    if not os.path.isdir(path): continue

    for f in os.listdir(path):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                Image.open(os.path.join(path, f)).verify()
            except Exception:
                print(f" Archivo corrupto: {cls}/{f}")
