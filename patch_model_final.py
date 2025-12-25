import json
import zipfile
from pathlib import Path

IN_FILE = Path("modelo_gestos.keras")
OUT_FILE = Path("modelo_gestos_fixed.keras")

def patch_config(cfg: dict) -> dict:
    def patch_layer(layer_obj: dict):
        if not isinstance(layer_obj, dict):
            return

        class_name = layer_obj.get("class_name")
        layer_cfg = layer_obj.get("config", {})

        if not isinstance(layer_cfg, dict):
            return

        # 1) Keras 3: RandomRotation 
        if class_name == "RandomRotation":
            layer_cfg.pop("value_range", None)

        # 2) Keras 3: Conv2D 
        if class_name == "Conv2D":
            layer_cfg.pop("batch_input_shape", None)

    # Modelos guardar capas aquí
    if isinstance(cfg, dict) and "config" in cfg and isinstance(cfg["config"], dict):
        layers = cfg["config"].get("layers")
        if isinstance(layers, list):
            for layer in layers:
                patch_layer(layer)

    return cfg

with zipfile.ZipFile(IN_FILE, "r") as zin:
    names = zin.namelist()
    if "config.json" not in names:
        raise RuntimeError(f"No encuentro config.json dentro de {IN_FILE}. Contenido: {names}")

    config = json.loads(zin.read("config.json").decode("utf-8"))
    config = patch_config(config)

    with zipfile.ZipFile(OUT_FILE, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name in names:
            if name == "config.json":
                zout.writestr("config.json", json.dumps(config, ensure_ascii=False))
            else:
                zout.writestr(name, zin.read(name))

print("OK ->", OUT_FILE)
