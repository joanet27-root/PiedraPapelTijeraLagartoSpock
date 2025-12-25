import cv2
import numpy as np
import json
import random
import time
import os
import keras

MODEL_PATH = "modelo_gestos_fixed.keras"
CAM_INDEX = 0

print("Cargando modelo...", MODEL_PATH)
model = keras.saving.load_model(MODEL_PATH, compile=False)
print("Modelo cargado.")

# Derivar IMG_SIZE y canales desde el modelo
in_shape = model.input_shape
if isinstance(in_shape, list):
    in_shape = in_shape[0]

_, H, W, C = in_shape
IMG_SIZE = (W, H)

print("Input esperado por el modelo:", model.input_shape)
print("IMG_SIZE ajustado a:", IMG_SIZE, "canales:", C)

with open("class_names.json", "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)
print("Clases:", CLASS_NAMES)

COUNTDOWN_STEP = 0.5
CAPTURE_WINDOW = 0.5

ICON_SIZE = 96
ICONS_DIR = "icons"

WINDOW_NAME = "Piedra Papel Tijera Lagarto Spock - IA"

WIN_RULES = {
    "piedra": ["tijera", "lagarto"],
    "papel": ["piedra", "spock"],
    "tijera": ["papel", "lagarto"],
    "lagarto": ["papel", "spock"],
    "spock": ["piedra", "tijera"],
}




# ===================== ICONOS =====================

def load_icons():
    """
    Carga los PNG de icons/ y los redimensiona a ICON_SIZE x ICON_SIZE.
    Espera archivos: piedra.png, papel.png, tijera.png, lagarto.png, spock.png
    """
    icons = {}
    name_to_file = {
        "piedra": "piedra.png",
        "papel": "papel.png",
        "tijera": "tijera.png",
        "lagarto": "lagarto.png",
        "spock": "spock.png",
    }

    for name, filename in name_to_file.items():
        path = os.path.join(ICONS_DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGRA
        if img is None:
            print(f"[AVISO] No se pudo cargar el icono: {path}")
            continue
        icon = cv2.resize(img, (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA)
        icons[name] = icon

    return icons


ICONS = load_icons()


def overlay_icon(frame, icon, x, y):
    """
    Dibuja 'icon' (BGRA) sobre 'frame' (BGR) en posición (x, y) con alpha.
    Si se sale por los bordes, recorta.
    """
    if icon is None:
        return

    h, w = frame.shape[:2]
    ih, iw = icon.shape[:2]

    if x >= w or y >= h:
        return
    x2 = min(x + iw, w)
    y2 = min(y + ih, h)
    icon_x2 = x2 - x
    icon_y2 = y2 - y

    roi = frame[y:y2, x:x2]

    icon_crop = icon[0:icon_y2, 0:icon_x2]

    if icon_crop.shape[2] == 4:
        icon_rgb = icon_crop[:, :, :3]
        alpha = icon_crop[:, :, 3] / 255.0
        alpha = alpha[..., None]
        roi[:] = roi * (1 - alpha) + icon_rgb * alpha
    else:
        roi[:] = icon_crop


# ===================== PREPROCESADO =====================

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

    # Adaptar a canales del modelo (C) y al tamaño (IMG_SIZE)
    if C == 1:
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, IMG_SIZE)
        roi_tensor = roi_resized.astype("float32")
        roi_tensor = roi_tensor[..., None]  # (H,W,1)
    else:
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        roi_resized = cv2.resize(roi_rgb, IMG_SIZE)
        roi_tensor = roi_resized.astype("float32")

    return roi_tensor, (x1, y1, x2, y2), frame_bgr


# ===================== LÓGICA JUEGO =====================

def elegir_cpu():
    return random.choice(CLASS_NAMES)


def decidir_ganador(jugador, cpu):
    if jugador == cpu:
        return "empate"
    if cpu in WIN_RULES.get(jugador, []):
        return "jugador"
    return "cpu"


# ===================== BOTONES (ratón) =====================

# coordenadas globales de los botones (se recalculan cada frame)
btn_restart = None  # (x1, y1, x2, y2)
btn_quit = None

# flags globales de clic
restart_requested = False
quit_requested = False

# estado global para que el callback sepa si tiene sentido pulsar
state = "idle"


def mouse_callback(event, x, y, flags, param):
    global restart_requested, quit_requested, btn_restart, btn_quit, state
    if event == cv2.EVENT_LBUTTONDOWN and state == "game_over":
        if btn_restart is not None:
            x1, y1, x2, y2 = btn_restart
            if x1 <= x <= x2 and y1 <= y <= y2:
                restart_requested = True
        if btn_quit is not None:
            x1, y1, x2, y2 = btn_quit
            if x1 <= x <= x2 and y1 <= y <= y2:
                quit_requested = True


def draw_buttons(frame, h, w, current_state):
    """
    Dibuja botones de Reiniciar y Salir si el estado es game_over.
    Devuelve las coordenadas actualizadas.
    """
    global btn_restart, btn_quit

    btn_restart = None
    btn_quit = None

    if current_state != "game_over":
        return

    font = cv2.FONT_HERSHEY_SIMPLEX

    btn_w = 180
    btn_h = 40
    y1 = h - 250
    y2 = y1 + btn_h

    # Botón REINICIAR (izquierda)
    x1 = 10
    x2 = x1 + btn_w
    btn_restart = (x1, y1, x2, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Reiniciar", (x1 + 20, y1 + 27),
                font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Botón SALIR (derecha)
    x1 = x2 + 20
    x2 = x1 + btn_w
    btn_quit = (x1, y1, x2, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, "Salir", (x1 + 55, y1 + 27),
                font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


# ===================== MAIN =====================

def main():
    global state, restart_requested, quit_requested

    # Elegir los puntos objetivo
    try:
        rounds_to_win = int(input("¿A cuántos puntos quieres jugar? (Ej: 3, 5, 7): ").strip())
        if rounds_to_win <= 0:
            print("Número invalido, se usara 3.")
            rounds_to_win = 3
    except Exception:
        print("Entrada no valida, se usara 3.")
        rounds_to_win = 3

    print(f"El primero que llegue a {rounds_to_win} puntos gana.")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    user_score = 0
    cpu_score = 0

    user_move = None
    cpu_move = None
    result_text = "Pulsa ESPACIO para empezar la ronda"

    # Estados del juego
    state = "idle"      # "idle", "countdown", "capturing", "game_over"
    countdown_value = 0
    countdown_start = 0.0
    capture_start = 0.0
    capture_preds = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        roi_tensor, (x1, y1, x2, y2), frame_bgr = preprocess_roi(frame_bgr)
        h, w, _ = frame_bgr.shape

        # ---------- LÓGICA DE ESTADOS ----------
        now = time.time()

        if state == "countdown":
            elapsed = now - countdown_start
            if elapsed >= COUNTDOWN_STEP:
                countdown_value -= 1
                countdown_start = now
                if countdown_value <= 0:
                    state = "capturing"
                    capture_start = now
                    capture_preds = []

        elif state == "capturing":
            if roi_tensor is not None:
                if now - capture_start <= CAPTURE_WINDOW:
                    input_tensor = np.expand_dims(roi_tensor, axis=0)
                    preds = model.predict(input_tensor, verbose=0)[0]
                    capture_preds.append(preds)
                else:
                    if capture_preds:
                        avg_pred = np.mean(capture_preds, axis=0)
                        idx = int(np.argmax(avg_pred))
                        user_move = CLASS_NAMES[idx]
                    else:
                        user_move = None

                    cpu_move = elegir_cpu()

                    if user_move is None:
                        result_text = "No se detecto gesto. Repite ronda."
                    else:
                        ganador = decidir_ganador(user_move, cpu_move)
                        if ganador == "jugador":
                            user_score += 1
                            result_text = "Ronda para el jugador."
                        elif ganador == "cpu":
                            cpu_score += 1
                            result_text = "Ronda para la CPU."
                        else:
                            result_text = "Empate."

                    if user_score >= rounds_to_win or cpu_score >= rounds_to_win:
                        state = "game_over"
                        if user_score > cpu_score:
                            result_text = f"Has ganado {user_score}-{cpu_score}! Haz clic en Reiniciar."
                        elif cpu_score > user_score:
                            result_text = f"Has perdido {user_score}-{cpu_score}! Haz clic en Reiniciar."
                        else:
                            result_text = f"Empate {user_score}-{cpu_score}. Haz clic en Reiniciar."
                    else:
                        state = "idle"
                        if user_move is None:
                            result_text += " | Pulsa espacio para repetir."
                        else:
                            result_text += " | Pulsa espacio para la siguiente ronda."

        # ---------- DIBUJO SOBRE EL FRAME ----------

        font = cv2.FONT_HERSHEY_SIMPLEX

        # ROI
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # USER texto
        cv2.putText(frame_bgr, "USER", (10, 40), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
        # icono USER
        if user_move in ICONS:
            overlay_icon(frame_bgr, ICONS[user_move], 10, 50)
        else:
            cv2.putText(frame_bgr, "?", (10, 100), font, 2, (0, 255, 255), 2, cv2.LINE_AA)

        # CPU texto
        cpu_text_x = w - 150
        cv2.putText(frame_bgr, "CPU", (cpu_text_x, 40), font, 1, (0, 165, 255), 2, cv2.LINE_AA)
        # icono CPU
        if cpu_move in ICONS:
            overlay_icon(frame_bgr, ICONS[cpu_move], w - ICON_SIZE - 10, 50)
        else:
            cv2.putText(frame_bgr, "?", (w - 100, 100), font, 2, (0, 165, 255), 2, cv2.LINE_AA)

        # Marcador
        score_text = f"Marcador: USER {user_score} - {cpu_score} CPU  (juegas a {rounds_to_win})"
        cv2.putText(frame_bgr, score_text, (10, h - 40), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Resultado
        cv2.putText(frame_bgr, f"Resultado: {result_text}", (10, h - 10),
                    font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Contador grande
        if state == "countdown" and countdown_value > 0:
            text = str(countdown_value)
            cv2.putText(frame_bgr, text, (w // 2 - 50, h // 2),
                        font, 3, (0, 0, 255), 5, cv2.LINE_AA)

        # Instrucciones
        if state == "idle":
            cv2.putText(frame_bgr, "Pulsa espacio para empezar la ronda",
                        (10, h - 70), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        elif state == "game_over":
            cv2.putText(frame_bgr, "Haz clic en Reiniciar o Salir",
                        (10, h - 70), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Botones (solo en game_over)
        draw_buttons(frame_bgr, h, w, state)

        cv2.imshow(WINDOW_NAME, frame_bgr)

        # ---------- RATON / TECLADO ----------

        # botones
        if quit_requested:
            break

        if restart_requested:
            # reiniciar estado del juego
            user_score = 0
            cpu_score = 0
            user_move = None
            cpu_move = None
            result_text = "Pulsa ESPACIO para empezar la ronda"
            state = "idle"
            restart_requested = False

        # teclas de respaldo (por si falla el raton)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if state == "idle" and key == ord(' '):
            state = "countdown"
            countdown_value = 3
            countdown_start = time.time()

        if state == "game_over" and key == ord('r'):
            user_score = 0
            cpu_score = 0
            user_move = None
            cpu_move = None
            result_text = "Pulsa ESPACIO para empezar la ronda"
            state = "idle"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
