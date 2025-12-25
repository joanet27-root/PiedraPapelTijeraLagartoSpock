# Rock Paper Scissors Lizard Spock — CNN Webcam Game

Real-time hand gesture recognition system for the game **Rock, Paper, Scissors, Lizard, Spock**, implemented using a **Convolutional Neural Network (CNN)** and a standard webcam.

The project combines computer vision, deep learning, and real-time interaction to classify hand gestures and play against a CPU opponent.

---

## Demo
▶️ **Video demo:**  
https://joanet27-root.github.io/assets/PiedraPapelTijeraLagartoSpock.mp4

---

## Features
- Real-time webcam inference using OpenCV
- CNN-based multi-class gesture classification (5 classes)
- Integrated data augmentation (rotation, zoom, flip)
- Interactive game logic with score tracking
- Visual feedback: ROI box, countdown, icons, and results

---

## Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## Project Structure

```text
.
├── main.py              # Training + evaluation pipeline
├── webcam_demo.py       # Real-time webcam game
├── model_cnn.py         # CNN architecture
├── load_dataset.py      # Dataset loader
├── train.py             # Training loop
├── evaluate.py          # Evaluation and metrics
├── utils.py             # Plotting utilities
├── captura_dataset.py   # (Optional) Dataset capture tool
├── inspect_dataset.py   # (Optional) Dataset inspection
├── icons/               # Gesture icons
└── assets/              # Demo media
