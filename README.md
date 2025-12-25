# Rock Paper Scissors Lizard Spock — CNN Webcam Game

Real-time hand gesture recognition system for the game **Rock, Paper, Scissors, Lizard, Spock**, implemented using a **Convolutional Neural Network (CNN)** and a standard webcam.

The project combines computer vision, deep learning, and interactive game logic to classify hand gestures in real time and play against a CPU opponent.

---

## Demo
**Video demo:**  
https://joanet27-root.github.io/assets/PiedraPapelTijeraLagartoSpock.mp4

---

## Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## Requirements
- Python 3.9+
- Webcam

---

## Installation

```bash
git clone https://github.com/joanet27-root/PiedraPapelTijeraLagartoSpock.git
cd PiedraPapelTijeraLagartoSpock
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Dataset Format

The dataset is not included in the repository.  
Expected directory structure:

```text
dataset/
├── piedra/
├── papel/
├── tijera/
├── lagarto/
└── spock/
```

- Images are RGB
- Automatically resized to **160 × 160**
- Labels are inferred from folder names

---

## Training the Model

```bash
python src/main.py
```

This script:
1. Loads and preprocesses the dataset  
2. Trains the CNN  
3. Evaluates performance on validation data  
4. Saves the trained model  

---

## Running the Webcam Game

```bash
python src/webcam_demo.py
```

### Controls
-**3, 5 or 7 in cmd** - Select the number of rounds
- **SPACE** — start a round  
- **Mouse click** — restart / exit (game over)  
- **Q** — quit application  

---

## Notes
- Ensure stable lighting and keep your hand inside the ROI box.
- If the webcam does not open, adjust `CAM_INDEX` in `src/webcam_demo.py`.

---

## Repository Structure
```text
.
├── src/                    # Core application and ML pipeline
│   ├── main.py
│   ├── webcam_demo.py
│   ├── model_cnn.py
│   ├── load_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── tools/                  # Auxiliary scripts
│   ├── captura_dataset.py
│   └── inspect_dataset.py
│
├── models/                 # Trained model (optional)
├── icons/                  # Gesture icons
├── assets/                 # Demo media
├── requirements.txt
└── README.md
```

