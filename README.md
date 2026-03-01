# EMNIST Alphanumeric Character Recognizer

A web application that recognizes handwritten characters (letters and digits) using a CNN model trained on the EMNIST Balanced dataset.

**Dataset:** [EMNIST on Kaggle](https://www.kaggle.com/datasets/crawford/emnist)

---

## Project Overview

This project was built to recognize handwritten alphanumeric characters using deep learning. The model was trained on the EMNIST Balanced dataset which contains 47 classes — digits 0–9 and case-merged letters. The full pipeline covers data preprocessing, CNN training, image processing, and a browser-based drawing interface.

---

## Model Architecture

A custom CNN was designed and trained from scratch using PyTorch:

- **Input:** 28×28 grayscale image (1 channel)
- **Block 1:** Conv2d(1→32) + BatchNorm + ReLU + MaxPool
- **Block 2:** Conv2d(32→64) + BatchNorm + ReLU + MaxPool
- **Block 3:** Conv2d(64→128) + BatchNorm + ReLU + MaxPool
- **Classifier:** Linear(128×3×3 → 256) + Dropout(0.4) + Linear(256 → 47)

**Training details:**
- Optimizer: Adam (lr=1e-3)
- Scheduler: StepLR (step=5, gamma=0.5)
- Epochs: 20
- Batch size: 128
- Hardware: Google Colab T4 GPU

---

## EMNIST Dataset — Problems & Solutions

### Problem 1: Wrong file path after download
The Kaggle EMNIST dataset did not extract into the expected `gzip/` subfolder. Files were already decompressed and located at `emnist_data/emnist_source_files/`. The IDX reader was updated to point to the correct path directly.

### Problem 2: Incorrect image orientation
EMNIST images are stored internally in a transposed and flipped format. When loaded raw, characters appeared rotated and mirrored. After experimenting with different combinations, the correct fix was a **90° counter-clockwise rotation** followed by a **horizontal flip**:

```python
X_train = np.rot90(X_train, k=3, axes=(1, 2)).copy()
X_train = X_train[:, :, ::-1].copy()
```

This was verified visually by plotting samples and confirming that labels matched the displayed characters.

---

## Image Processing Pipeline

### Challenge: Input must match EMNIST format
EMNIST uses **white object on black background** in grayscale. User drawings on the canvas could have any color combination, so a normalization pipeline was implemented:

**Frontend (JavaScript):**
- The drawing canvas is 400×400px but the model expects 28×28
- A bounding box is computed around the drawn character
- The character is cropped, padded, centered, and resized to exactly 28×28 using an offscreen canvas:

```javascript
offCtx.drawImage(canvas, cx - size/2, cy - size/2, size, size, 0, 0, 28, 28);
```

- This ensures characters drawn near edges or corners are still correctly recognized

**Backend (Python):**
- Image is converted to grayscale using `PIL.Image.convert('L')`
- Background detection uses the **75th percentile** of pixel values — if the majority of the image is light, the image is inverted:

```python
bg_val = float(np.percentile(arr, 75))
if bg_val > 0.5:
    arr = 1.0 - arr
```

- This is more robust than checking corner pixels, which failed when characters were drawn near the edges

---

## Setup & Run Locally

```bash
git clone https://github.com/nuranium92/Character-Detection.git
cd "Character Detection"

python -m venv venv
venv\Scripts\activate

pip install -r backend/requirements.txt

cd backend
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## Model Training

To retrain the model, open `model/emnist_train.ipynb` in Google Colab, enable T4 GPU, and run all cells. The notebook will download the dataset via Kaggle API, preprocess it, train the CNN, and export `model.pth` and `label_map.json`.

---

*by nuranium*