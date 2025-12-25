import cv2
import os
import time

IMG_SIZE = (160, 160)

DATASET_DIR = "../dataset"  # mismo que en load_dataset.py

GESTURE_KEYS = {
    ord("1"): "piedra",
    ord("2"): "papel",
    ord("3"): "tijera",
    ord("4"): "lagarto",
    ord("5"): "spock",
}

def ensure_dirs():
    for cls in GESTURE_KEYS.values():
        path = os.path.join(DATASET_DIR, cls)
        os.makedirs(path, exist_ok=True)

def preprocess_roi(frame_bgr):
    frame_bgr = cv2.flip(frame_bgr, 1)
    h, w, _ = frame_bgr.shape
    box_size = min(h, w) // 2
    x2 = w - 20
    x1 = x2 - box_size
    y1 = (h - box_size) // 2
    y2 = y1 + box_size

    roi_bgr = frame_bgr[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        return None, (x1, y1, x2, y2), frame_bgr

    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, IMG_SIZE)
    return roi_resized, (x1, y1, x2, y2), frame_bgr

def main():
    ensure_dirs()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    print("Pulsa 1=piedra, 2=papel, 3=tijera, 4=lagarto, 5=spock para guardar imagen.")
    print("Pulsa q para salir.")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        roi_img, (x1, y1, x2, y2), frame_bgr = preprocess_roi(frame_bgr)

        if roi_img is not None:
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Captura dataset", frame_bgr)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key in GESTURE_KEYS and roi_img is not None:
            cls = GESTURE_KEYS[key]
            timestamp = int(time.time() * 1000)
            filename = os.path.join(DATASET_DIR, cls, f"{cls}_{timestamp}.jpg")
            # guardamos en RGB; image_dataset_from_directory los lee bien
            roi_bgr_save = cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, roi_bgr_save)
            print(f"Guardada {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
