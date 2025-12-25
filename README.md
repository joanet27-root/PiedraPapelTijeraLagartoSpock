# Rock Paper Scissors Lizard Spock — CNN Webcam Game

Real-time hand gesture recognition system for the game **Rock, Paper, Scissors, Lizard, Spock**, implemented using a **Convolutional Neural Network (CNN)** and a standard webcam.

The project integrates computer vision, deep learning, and interactive game logic to perform real-time gesture classification.

---

## Demo
**Video demo:**  
https://joanet27-root.github.io/assets/PiedraPapelTijeraLagartoSpock.mp4

---

## Project Structure

```text
.
├── src/                    # Core application and ML pipeline
│   ├── main.py             # Training + evaluation
│   ├── webcam_demo.py      # Real-time webcam game
│   ├── model_cnn.py        # CNN architecture
│   ├── load_dataset.py     # Dataset loading
│   ├── train.py            # Training logic
│   ├── evaluate.py         # Evaluation metrics
│   └── utils.py            # Plotting utilities
│
├── tools/                  # Auxiliary scripts
│   ├── captura_dataset.py  # Dataset capture tool
│   └── inspect_dataset.py  # Dataset inspection
│
├── models/                 # Trained models (optional)
│   └── cnn_gesture_model.keras
│
├── icons/                  # Gesture icons used in the game
├── assets/                 # Media (video demo)
├── class_names.json
├── requirements.txt
└── README.md
