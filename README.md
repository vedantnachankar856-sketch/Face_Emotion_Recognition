# 😊 Face Emotion Recognition App

A Streamlit web application that detects faces in images and predicts emotions using a deep learning CNN model trained on the **FER2013** dataset.

---

## 🎯 Detected Emotions

| Emotion | Emoji |
|---------|-------|
| Angry | 😠 |
| Disgust | 🤢 |
| Fear | 😨 |
| Happy | 😊 |
| Neutral | 😐 |
| Sad | 😢 |
| Surprise | 😲 |

---

## 🧠 Model Architecture

```
Conv2D(32) → Conv2D(64) → MaxPool → Dropout(0.1)
Conv2D(128)             → MaxPool → Dropout(0.1)
Conv2D(256)             → MaxPool → Dropout(0.1)
Flatten → Dense(512) → Dropout(0.2) → Dense(7, softmax)
```

- Input: 48×48 grayscale images  
- Output: 7-class softmax probabilities  
- Loss: Categorical Crossentropy | Optimizer: Adam

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/face-emotion-recognition.git
cd face-emotion-recognition
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (or use a pre-trained one)

Open `Face_Emotion_Recognition_using_Deep_Learning_Solution.ipynb` in Kaggle or Google Colab, run all cells, and download the saved `model_file.h5`.

Place `model_file.h5` in the root of the project directory.

> **Dataset:** [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
face-emotion-recognition/
│
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
├── model_file.h5                   # Trained model (add after training)
├── Face_Emotion_Recognition_using_Deep_Learning_Solution.ipynb
└── README.md
```

---

## 🖥️ App Features

- **Upload Image tab** — upload any JPG/PNG photo; detects all faces and shows emotion probabilities for each one
- **Webcam tab** — take a live photo from your device camera
- Annotated output image with bounding boxes and labels
- Detailed confidence bars for all 7 emotions per face

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub (include `model_file.h5`)
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New app** → select your repo → set `app.py` as the main file
4. Click **Deploy**

> ⚠️ If `model_file.h5` is larger than 100 MB, use [Git LFS](https://git-lfs.github.com/) to store it.

---

## 📊 Dataset

**FER2013** — Facial Expression Recognition 2013  
- 35,887 grayscale 48×48 face images  
- 7 emotion categories  
- Source: [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

---

## 📄 License

MIT License — feel free to use and modify.
