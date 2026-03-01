# EMNIST Alphanumeric Character Recognizer

A web application that recognizes handwritten characters (letters and digits) using a CNN model trained on the EMNIST Balanced dataset.

**Live Demo:** [Railway URL]  
**Dataset:** [EMNIST on Kaggle](https://www.kaggle.com/datasets/crawford/emnist)

---

## Features

- Handwritten character recognition (47 classes: digits 0-9 + letters)
- Interactive drawing canvas
- Top 3 predictions with confidence bar chart
- Flask backend + vanilla JS frontend

---

## Project Structure

```
Character Detection/
├── model/
│   ├── train.py
│   ├── model.pth
│   └── label_map.json
├── backend/
│   ├── app.py
│   ├── predict.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── canvas.js
├── Dockerfile
└── README.md
```

---

## Problems Encountered & Solutions

### 1. TensorFlow Installation Failed
**Problem:** `pip install tensorflow` failed with "No matching distribution found".  
**Cause:** Python 3.13 is not supported by TensorFlow.  
**Solution:** Switched to **PyTorch** instead of TensorFlow. PyTorch supports Python 3.13 and works on Windows.

---

### 2. EMNIST Dataset File Path Error
**Problem:** `FileNotFoundError: emnist_data/gzip/emnist-balanced-train-images-idx3-ubyte.gz`  
**Cause:** The Kaggle EMNIST dataset came pre-extracted. Files were inside `emnist_data/emnist_source_files/` and already decompressed.  
**Solution:** Updated the base path to `emnist_data/emnist_source_files/` and removed the `.gz` extraction step.

---

### 3. EMNIST Image Orientation Problem
**Problem:** Characters appeared mirrored and rotated after loading the dataset.  
**Cause:** EMNIST stores images transposed and flipped compared to standard orientation.  
**Solution:** Applied `np.rot90(..., k=3)` (90° counter-clockwise) followed by a horizontal flip `[:, :, ::-1]` to restore correct orientation before training.

---

### 4. Model Could Not Find `model.pth`
**Problem:** `FileNotFoundError: ../model/model.pth` when running Flask backend.  
**Cause:** Relative paths don't work reliably depending on the working directory.  
**Solution:** Used `os.path.abspath(__file__)` to construct absolute paths dynamically:
```python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'model', 'model.pth')
```

---

### 5. Flask `localhost:5000` Returned 404
**Problem:** Navigating to `localhost:5000` showed "Not Found".  
**Cause:** Flask had no route defined for `/`.  
**Solution:** Added a route to serve `index.html` from the frontend directory:
```python
@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')
```

---

### 6. Flask OSError on Windows with `host='0.0.0.0'`
**Problem:** `OSError: [WinError 10038] An operation was attempted on something that is not a socket`  
**Cause:** Flask's debug mode reloader conflicts with `host='0.0.0.0'` on Windows.  
**Solution:** Set `debug=False` and used `os.environ.get('PORT', 5000)` for Railway compatibility:
```python
app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
```

---

### 7. Prediction Fails for Characters Drawn Near Edges
**Problem:** Characters drawn near canvas edges were predicted incorrectly.  
**Cause:** The full canvas was sent to the model — the character occupied only a small portion of the 28x28 image.  
**Solution:** Implemented bounding box detection in JavaScript. The drawn character is cropped, centered, padded, and resized to exactly 28x28 before sending to the backend:
```javascript
offCtx.drawImage(canvas, cx - size/2, cy - size/2, size, size, 0, 0, 28, 28);
```

---

### 8. Background Inversion Logic Was Unreliable
**Problem:** The backend used 4 corner pixels to detect whether to invert the image. Characters drawn near corners caused incorrect inversion.  
**Solution:** Replaced corner-pixel detection with the **75th percentile** of pixel values:
```python
bg_val = float(np.percentile(arr, 75))
if bg_val > 0.5:
    arr = 1.0 - arr
```

---

### 9. Git Push Rejected — No Upstream Branch
**Problem:** `fatal: The current branch master has no upstream branch`  
**Cause:** Local branch was `master` but GitHub expected `main`.  
**Solution:**
```bash
git branch -M main
git push -u origin main
```

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

Model was trained on Google Colab (T4 GPU) using the EMNIST Balanced dataset.  
To retrain: open `model/emnist_train.ipynb` in Google Colab and run all cells.

**Architecture:** CNN with 3 conv blocks (32→64→128 filters) + BatchNorm + Dropout  
**Dataset:** EMNIST Balanced — 47 classes, ~112,000 training samples  
**Training:** 20 epochs, Adam optimizer, StepLR scheduler  

---

## Deployment

Deployed on Railway via Docker. Push to `main` triggers automatic redeploy.

---

*by nuranium*
