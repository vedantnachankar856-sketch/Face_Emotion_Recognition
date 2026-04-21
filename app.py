import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="Face Emotion Recognition",
    page_icon="😊",
    layout="centered"
)

# Emotion labels (FER2013 dataset classes)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Emoji mapping
EMOTION_EMOJI = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😊',
    'Neutral': '😐',
    'Sad': '😢',
    'Surprise': '😲'
}

# Color mapping for progress bars
EMOTION_COLOR = {
    'Angry': '#FF4B4B',
    'Disgust': '#8B4513',
    'Fear': '#9370DB',
    'Happy': '#FFD700',
    'Neutral': '#808080',
    'Sad': '#4169E1',
    'Surprise': '#FF69B4'
}

@st.cache_resource
@st.cache_resource
def load_model():
    try:
        import tf_keras as keras
        model = keras.models.load_model('model_file.h5')
        return model
    except Exception as e:
        st.warning(f"Model not loaded: {e}")
        return None

@st.cache_resource
def load_face_cascade():
    """Load OpenCV Haar Cascade for face detection."""
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face_img):
    """Preprocess a face crop for model prediction."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    input_data = np.reshape(normalized, (1, 48, 48, 1))
    return input_data

def predict_emotion(model, face_crop):
    """Predict emotion probabilities for a face crop."""
    input_data = preprocess_face(face_crop)
    predictions = model.predict(input_data, verbose=0)[0]
    return predictions

def detect_and_predict(image_array, model, face_cascade):
    """Detect faces and predict emotions, return annotated image + results."""
    img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    results = []
    annotated = image_array.copy()

    for i, (x, y, w, h) in enumerate(faces):
        face_crop = img_bgr[y:y+h, x:x+w]
        probs = predict_emotion(model, face_crop)
        top_idx = np.argmax(probs)
        top_emotion = EMOTIONS[top_idx]
        confidence = probs[top_idx]

        # Draw rectangle and label
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{top_emotion} ({confidence:.0%})"
        cv2.putText(annotated, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        results.append({
            'face_id': i + 1,
            'emotion': top_emotion,
            'confidence': confidence,
            'probabilities': dict(zip(EMOTIONS, probs))
        })

    return annotated, results, len(faces)

# ── UI ──────────────────────────────────────────────────────────────────────

st.title("😊 Face Emotion Recognition")
st.markdown("Detect faces and recognize emotions using a Deep Learning model trained on the FER2013 dataset.")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model:** CNN trained on FER2013  
    **7 Emotions detected:**
    """)
    for e, emoji in EMOTION_EMOJI.items():
        st.markdown(f"- {emoji} {e}")

    st.divider()
    st.markdown("**Dataset:** [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)")
    st.markdown("**Architecture:** Conv2D → MaxPool → Dropout (×3) → Dense(512) → Softmax(7)")

# Load resources
model = load_model()
face_cascade = load_face_cascade()

if model is None:
    st.warning("""
    ⚠️ **Model file not found.**

    To use this app:
    1. Train the model using the notebook.
    2. Place `model_file.h5` in the same directory as `app.py`.
    3. Restart the app.

    The app will still show the UI and face detection boxes even without the model.
    """)

# Input method
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Use Webcam"])

# ── Tab 1: Upload ──
with tab1:
    uploaded_file = st.file_uploader(
        "Upload a photo containing one or more faces",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        if model is not None:
            with st.spinner("Detecting faces and predicting emotions..."):
                annotated, results, num_faces = detect_and_predict(image_array, model, face_cascade)

            with col2:
                st.subheader(f"Detected: {num_faces} face(s)")
                st.image(annotated, use_container_width=True)

            if num_faces == 0:
                st.info("No faces detected. Try a clearer, well-lit photo.")
            else:
                st.divider()
                st.subheader("Emotion Analysis")
                for r in results:
                    with st.expander(f"Face #{r['face_id']}  —  {EMOTION_EMOJI[r['emotion']]} **{r['emotion']}** ({r['confidence']:.1%})", expanded=True):
                        for emotion, prob in sorted(r['probabilities'].items(), key=lambda x: -x[1]):
                            col_a, col_b = st.columns([3, 1])
                            col_a.progress(float(prob), text=f"{EMOTION_EMOJI[emotion]} {emotion}")
                            col_b.markdown(f"**{prob:.1%}**")
        else:
            with col2:
                st.subheader("Face Detection Only")
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                gray_cv = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_cv, scaleFactor=1.1, minNeighbors=5)
                annotated = image_array.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
                st.image(annotated, use_container_width=True)
                st.info(f"Detected {len(faces)} face(s). Load `model_file.h5` to see emotions.")

# ── Tab 2: Webcam ──
with tab2:
    st.info("📷 Capture a photo using your device camera.")
    camera_photo = st.camera_input("Take a photo")

    if camera_photo:
        image = Image.open(camera_photo).convert("RGB")
        image_array = np.array(image)

        if model is not None:
            with st.spinner("Analyzing..."):
                annotated, results, num_faces = detect_and_predict(image_array, model, face_cascade)

            st.image(annotated, caption=f"{num_faces} face(s) detected", use_container_width=True)

            if num_faces == 0:
                st.info("No faces detected. Make sure your face is clearly visible.")
            else:
                for r in results:
                    st.markdown(f"### Face #{r['face_id']}  {EMOTION_EMOJI[r['emotion']]} {r['emotion']} — {r['confidence']:.1%}")
                    for emotion, prob in sorted(r['probabilities'].items(), key=lambda x: -x[1]):
                        st.progress(float(prob), text=f"{EMOTION_EMOJI[emotion]} {emotion}  {prob:.1%}")
        else:
            st.image(image, use_container_width=True)
            st.warning("Model not loaded. Cannot predict emotions.")
git add app.py
git commit -m "fix: add except block to load_model try statement"
git push
st.divider()
st.caption("Built with Streamlit · CNN trained on FER2013 · OpenCV face detection")
