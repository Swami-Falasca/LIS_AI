# app.py per Streamlit
import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Titolo
st.title("🎨 App Disegno con Riconoscimento Gestuale")

# Sidebar
st.sidebar.header("Controlli")
color = st.sidebar.color_picker("Scegli colore", "#000000")
brush_size = st.sidebar.slider("Dimensione pennello", 1, 20, 5)
clear_btn = st.sidebar.button("Pulisci Canvas")

# Canvas HTML/JavaScript
st.markdown("""
<div style="border:2px solid #000; width:800px; height:600px;">
    <canvas id="myCanvas" width="800" height="600" style="border:1px solid #000000;"></canvas>
</div>

<script>
const canvas = document.getElementById('myCanvas');
const ctx = canvas.getContext('2d');
let painting = false;

canvas.addEventListener('mousedown', startPosition);
canvas.addEventListener('mouseup', finishedPosition);
canvas.addEventListener('mousemove', draw);

function startPosition(e) {
    painting = true;
    draw(e);
}

function finishedPosition() {
    painting = false;
    ctx.beginPath();
}

function draw(e) {
    if(!painting) return;
    ctx.lineWidth = %s;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '%s';
    
    ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
}
</script>
""" % (brush_size, color), unsafe_allow_html=True)

# Carica modello AI
@st.cache_resource
def load_model():
    model_dict = pickle.load(open('./model.p', 'rb'))
    return model_dict['model']

model = load_model()

# Webcam con riconoscimento gesti
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.model = model
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Processa con MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            data_aux = []
            
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x)
                data_aux.append(landmark.y)
            
            if len(data_aux) == 42:
                prediction = self.model.predict([np.asarray(data_aux)])
                gesture = str(prediction[0])
                
                # Mostra gesto riconosciuto
                cv2.putText(img, f"Gesto: {gesture}", (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img

# Stream webcam
st.header("Webcam con Riconoscimento Gestuale")
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# Installazioni necessarie:
# pip install streamlit streamlit-webrtc opencv-python mediapipe numpy pillow av