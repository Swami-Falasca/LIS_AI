import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque


def flip_hand_landmarks(landmarks):
    """Ribalta i landmarks della mano (destra ↔ sinistra)"""
    if len(landmarks) == 42:
        flipped = []
        for i in range(0, len(landmarks), 2):
            x = 1 - landmarks[i]
            y = landmarks[i + 1]
            flipped.extend([x, y])
        return np.array(flipped)
    return landmarks

# Inizializza webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Classificatore ibrido
class HybridGestureRecognizer:
    def __init__(self):
        # Carica modello statico
        self.static_model = pickle.load(open('./model.p','rb'))['model']
        
        # Prova a caricare modello dinamico
        try:
            with open('./dynamic_model.p', 'rb') as f:
                dynamic_data = pickle.load(f)
                self.dynamic_model = dynamic_data['model']
                self.dynamic_label_map = dynamic_data['label_map']
                self.has_dynamic_model = True
        except:
            self.has_dynamic_model = False
        
        # Buffer per sequenze
        self.sequence_buffer = deque(maxlen=30)
        self.current_prediction = ""
    
    def predict(self, landmarks):
        # Predizione statica
        static_pred = self.static_model.predict([landmarks])[0]
        
        # Se esiste modello dinamico
        if self.has_dynamic_model:
            self.sequence_buffer.append(landmarks)
            
            if len(self.sequence_buffer) == 30:
                has_movement = analyze_movement(self.sequence_buffer)
                
                if has_movement:
                    sequence_array = np.array(self.sequence_buffer).reshape(1, 30, -1)
                    dynamic_pred_proba = self.dynamic_model.predict(sequence_array, verbose=0)
                    dynamic_pred_idx = np.argmax(dynamic_pred_proba[0])
                    dynamic_pred = self.dynamic_label_map[dynamic_pred_idx]
                    
                    confidence = np.max(dynamic_pred_proba[0])
                    
                    if confidence > 0.8 and dynamic_pred != static_pred:
                        self.current_prediction = dynamic_pred
                    else:
                        self.current_prediction = static_pred
                else:
                    self.current_prediction = static_pred
                
                self.sequence_buffer.clear()
            else:
                self.current_prediction = static_pred
                return static_pred
        else:
            self.current_prediction = static_pred
            self.sequence_buffer.clear()
            return static_pred

        return self.current_prediction

# Funzione per rilevare movimento
def analyze_movement(sequence_buffer):
    if len(sequence_buffer) < 2:
        return False
    
    sequence_array = np.array(list(sequence_buffer))
    position_variance = np.var(sequence_array, axis=0)
    x_variance = position_variance[::2]
    mean_variance = np.mean(x_variance)
    
    return mean_variance > 0.001

# Inizializza riconoscitore
recognizer = HybridGestureRecognizer()

# Variabili per visualizzazione
last_display_char = ""
display_counter = 0
HOLD_FRAMES = 40

last_dynamic_char = ""
dynamic_lock_frames = 60
dynamic_lock_counter = 0

# Loop principale
while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape  
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Se ci sono mani
    if results.multi_hand_landmarks:
        hand_predictions = []
        
        # Processa ogni mano
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Disegna landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Estrai landmarks
            data_aux = []
            x_ = []
            y_ = []
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)
            
            # Bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10
            
            # Predizione
            if len(data_aux) == 42:
                # Prova mano originale
                predicted_original = recognizer.predict(np.asarray(data_aux))
                
                # Prova mano ribaltata
                flipped_landmarks = flip_hand_landmarks(data_aux)
                predicted_flipped = recognizer.predict(flipped_landmarks)
                
                # Usa predizione originale
                predicted_character = predicted_original
                
                # Memorizza
                hand_predictions.append({
                    'hand_idx': hand_idx,
                    'bbox': (x1, y1, x2, y2),
                    'prediction': predicted_character,
                })
        
        # Prendi prima mano per visualizzazione
        if hand_predictions:
            hand_info = hand_predictions[0]
            x1, y1, x2, y2 = hand_info['bbox']
            predicted_character = hand_info['prediction']
            
            # Logica visualizzazione
            if dynamic_lock_counter > 0:
                dynamic_lock_counter -= 1
                final_char = last_dynamic_char
            elif predicted_character in ['SJ', 'Z', 'z', 'SS', 'SG']:
                last_dynamic_char = predicted_character
                dynamic_lock_counter = dynamic_lock_frames
                last_display_char = predicted_character
                display_counter = HOLD_FRAMES
                final_char = predicted_character
            elif predicted_character != last_display_char:
                last_display_char = predicted_character
                display_counter = HOLD_FRAMES
                final_char = predicted_character
            else:
                final_char = last_display_char
            
            # Visualizza
            if display_counter > 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, final_char, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 2, cv2.LINE_AA)
                display_counter -= 1

    # Mostra frame
    cv2.imshow('Riconoscimento LIS', frame)

    # Uscita
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()