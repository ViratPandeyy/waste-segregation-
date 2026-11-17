import json
from pathlib import Path
import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "waste_mobilenetv2.keras"
CLASS_PATH = BASE / "models" / "class_names.json"

st.set_page_config(page_title="Smart AI Waste Segregation", page_icon="♻️", layout="wide")
st.title("♻️ Smart AI Waste Segregation")
st.write("Live camera classification for plastic, glass, paper and organic waste.")
st.info("Phase 1 classifies the central camera area. It does not draw separate boxes around multiple objects.")

@st.cache_resource
def load_model():
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = json.loads(CLASS_PATH.read_text())
    return model, classes

model = None
classes = ["glass", "organic", "paper", "plastic"]
if MODEL_PATH.exists() and CLASS_PATH.exists():
    try:
        model, classes = load_model()
        st.success("Trained model loaded.")
    except Exception as e:
        st.error(f"Could not load model: {e}")
else:
    st.warning("No trained model found yet. Add training images to the dataset folders, then run train_mac.command or follow README.md.")


def classify(frame):
    if model is None:
        return "Model not trained", 0.0
    image = cv2.resize(frame, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float32")
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image, verbose=0)[0]
    idx = int(np.argmax(preds))
    return str(classes[idx]), float(preds[idx])

class WasteTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        x1, y1, x2, y2 = int(w*.25), int(h*.20), int(w*.75), int(h*.80)
        roi = img[y1:y2, x1:x2]
        label, confidence = classify(roi)
        cv2.rectangle(img, (x1,y1), (x2,y2), (40,220,80), 2)
        text = label if label == "Model not trained" else f"{label.upper()}  {confidence*100:.1f}%"
        cv2.rectangle(img, (10,10), (min(w-10, 420), 52), (20,20,20), -1)
        cv2.putText(img, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
        return img

st.subheader("Live camera")
webrtc_streamer(key="waste-camera", video_transformer_factory=WasteTransformer, media_stream_constraints={"video": True, "audio": False}, async_processing=True)

st.divider()
st.subheader("Test with an image")
file = st.file_uploader("Choose a waste image", type=["jpg", "jpeg", "png"])
if file:
    pil = Image.open(file).convert("RGB")
    st.image(pil, caption="Uploaded image", width=350)
    if model is not None:
        arr = np.array(pil.resize((224,224))).astype("float32")[None, ...]
        p = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(p))
        st.success(f"Prediction: **{classes[idx]}** ({float(p[idx])*100:.1f}% confidence)")
