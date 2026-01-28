import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque
import os

#se non va provare 1, 2...
#cap = cv2.VideoCapture(0)
#dovrebbe essere una nuova versione
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  

#Esattamente come in create dataset
#Detect and draw landmarks sulle immagini per la classificazione
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# static_image_mode=False per video, True solo per immagini statiche
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

labels_dict = {0: 'a', 1: 'b', 2: 'c'}

#DINAMICO
sequence_buffer = deque(maxlen=30)  # Buffer per 30 frame
dynamic_mode = False  # True per attivare riconoscimento dinamico

movement_detected = False
movement_frames = 0
MOVEMENT_THRESHOLD = 0.0005  # Soglia più bassa per rilevamento precoce


class HybridGestureRecognizer:
    def __init__(self, dynamic_model_path=None):
        # Carica modello statico ORIGINALE
        self.static_model = pickle.load(open('./model.p','rb'))['model']
        
        # Cerca modello dinamico se non specificato
        if dynamic_model_path is None:
            # Cerca in dynamic_models/
            if os.path.exists('./dynamic_models'):
                model_files = [f for f in os.listdir('./dynamic_models') 
                             if f.endswith('_model.p')]
                if model_files:
                    dynamic_model_path = f'./dynamic_models/{model_files[0]}'
            # Altrimenti cerca nella root
            elif os.path.exists('./dynamic_model.p'):
                dynamic_model_path = './dynamic_model.p'
        
        # Prova a caricare modello dinamico
        if dynamic_model_path and os.path.exists(dynamic_model_path):
            try:
                with open(dynamic_model_path, 'rb') as f:
                    dynamic_data = pickle.load(f)
                    self.dynamic_model = dynamic_data['model']
                    self.dynamic_label_map = dynamic_data['label_map']
                    self.model_type = dynamic_data.get('model_name', 'unknown')
                    self.input_shape = dynamic_data.get('input_shape', None)
                    self.has_dynamic_model = True
                    
                    print(f"✅ Modello dinamico caricato: {os.path.basename(dynamic_model_path)}")
                    print(f"   Tipo: {self.model_type}")
                    print(f"   Lettere: {list(self.dynamic_label_map.values())}")
                    
                    # Debug: mostra shape atteso
                    if self.input_shape:
                        print(f"   Input shape atteso: {self.input_shape}")
                    
            except Exception as e:
                print(f"❌ Errore caricamento {dynamic_model_path}: {e}")
                self.has_dynamic_model = False
        else:
            self.has_dynamic_model = False
            print("⚠️  Nessun modello dinamico trovato")
        
        # Buffer per sequenze
        self.sequence_buffer = deque(maxlen=30)
        self.last_dynamic_time = 0
        self.dynamic_cooldown = 15  # Attesa minima tra predizioni dinamiche
    
    def predict(self, landmarks):
        """Predice usando entrambi i modelli - FUNZIONA CON TUTTI I TIPI"""
        
        # 1. Predizione statica
        static_pred = self.static_model.predict([landmarks])[0]
        
        # 2. Se abbiamo modello dinamico
        if self.has_dynamic_model:
            # Aggiungi alla sequenza
            self.sequence_buffer.append(landmarks)
            
            # Controlla cooldown per evitare predizioni troppo ravvicinate
            self.last_dynamic_time += 1

            # Quando abbiamo 30 frame
            if len(self.sequence_buffer) == 30:
                has_movement, variance = analyze_movement(self.sequence_buffer)
                
                if has_movement and self.last_dynamic_time >= self.dynamic_cooldown:
                    sequence_array = np.array(self.sequence_buffer)
                    
                    # ====== GESTIONE TUTTI I TIPI DI MODELLO ======
                    
                    # A) RANDOM FOREST / XGBoost / SVM / MLP
                    #    (modelli scikit-learn con predict_proba)
                    if hasattr(self.dynamic_model, 'predict_proba'):
                        # Appiattisci: (30, 42) -> (1260,)
                        sequence_flat = sequence_array.flatten().reshape(1, -1)
                        pred_proba = self.dynamic_model.predict_proba(sequence_flat)[0]
                    
                    # B) MODELLI KERAS (LSTM/GRU/CNN)
                    #    (hanno predict ma non predict_proba)
                    elif hasattr(self.dynamic_model, 'predict'):
                        # Mantieni shape: (1, 30, 42)
                        sequence_reshaped = sequence_array.reshape(1, 30, 42)
                        pred_proba = self.dynamic_model.predict(sequence_reshaped, verbose=0)[0]
                    
                    # C) MODELLO SCONOSCIUTO
                    else:
                        print("⚠️  Tipo modello sconosciuto, uso statico")
                        self.sequence_buffer.clear()
                        return static_pred
                    
                    # Predizione finale
                    pred_idx = np.argmax(pred_proba)
                    dynamic_pred = self.dynamic_label_map[pred_idx]
                    confidence = pred_proba[pred_idx]
                    
                    # Log dettagliato
                    print(f"[{self.model_type}] {dynamic_pred} ({confidence:.1%})")
                    
                    if confidence > 0.85 and dynamic_pred!='S':  # Aumenta da 0.8 a 0.85 o più
                        print(f"🔄 Dinamico attivato ({confidence:.1%})")
                        self.sequence_buffer.clear()
                        return dynamic_pred
                    elif confidence > 0.90:
                        print(f"🔄 Dinamico attivato ({confidence:.1%})")
                        self.sequence_buffer.clear()
                        return dynamic_pred
                    else:
                        print(f"   Confidence bassa o stessa lettera")
                
                else:
                    print(f"[Statico] Nessun movimento")
                
                self.sequence_buffer.clear()
                return static_pred
            else:
                # Sequenza non completa
                seq_len = len(self.sequence_buffer)
                if seq_len % 10 == 0:  # Log ogni 10 frame
                    print(f"[Accumulo] {seq_len}/30 frame")
                return static_pred
        else:
            # Nessun modello dinamico
            return static_pred

#Carico il modello
#model_dict = pickle.load(open('./model.p','rb'))
#model = model_dict['model']

recognizer = HybridGestureRecognizer(
    dynamic_model_path='./dynamic_models/gru_model.p'
)

last_display_char = ""
display_counter = 0
HOLD_FRAMES = 40  # Mostra per 40 frame

last_dynamic_char = ""
dynamic_lock_frames = 40  # Blocca per 40 frame dopo una dinamica
dynamic_lock_counter = 0

#Riconosce se c'è movimento nella sequenza
def analyze_movement(sequence_buffer):
    if len(sequence_buffer) < 2:
        return False
    
    #converte in array numpy
    sequence_array = np.array(list(sequence_buffer))
    
    #calcola varianza delle posizioni nel tempo
    #(se le dita si muovono, la varianza sarà alta)
    position_variance = np.var(sequence_array, axis=0)
    
    #se la varianza media supera una soglia, c'è movimento
    movement_threshold = 0.005  # Più alta = meno falsi positivi  
    
    # Filtro migliore: controlla sia X che Y
    xy_variance = np.var(sequence_array, axis=0)
    mean_variance = np.mean(xy_variance)  # Media di TUTTE le features
    
    # Controlla anche il movimento specifico delle dita
    finger_variance = np.var(sequence_array[:, [20,21,22,23]], axis=0)  # Dita
    finger_movement = np.mean(finger_variance)
    
    return (mean_variance > movement_threshold and finger_movement > 0.0005), mean_variance

while True:

    ret, frame = cap.read()
    if not ret:
        break

    #perché restituirebbe 3 valori (il terzo è canali)
    H, W, _ = frame.shape  

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Detect all the landmarks in this image
    results = hands.process(frame_rgb)

        #Itera nel risultato
    if results.multi_hand_landmarks:
        # 1. Prima disegniamo TUTTE le mani
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                    frame, #image to draw
                    hand_landmarks, #model output
                    mp_hands.HAND_CONNECTIONS, #hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Prendiamo solo la PRIMA mano per la predizione
        # (per evitare problemi con 84 features invece di 42)
        first_hand = results.multi_hand_landmarks[0]
        data_aux = []
        x_ = []
        y_ = []
        
        #Creiamo array da landmarks della PRIMA mano
        for i in range(len(first_hand.landmark)):
            #x,y,z posizioni dei landmarks
            #a noi servono solo x e y
            #print(hand_landmarks.landmark[i])
            x = first_hand.landmark[i].x
            y = first_hand.landmark[i].y
            data_aux.append(x)
            data_aux.append(y)
            x_.append(x)  # Solo prima mano
            y_.append(y)  # Solo prima mano

      
        #questi sono i bordi del rettangolo che contengono la mano
        x1 = int(min(x_) * W) - 10  # -10 per margine
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        # Usa il riconoscitore IBRIDO (statico + dinamico automaticamente)
        if len(data_aux) == 42:

            
            predicted_character = recognizer.predict(np.asarray(data_aux))
            

            # 1. Se abbiamo un blocco attivo per dinamica precedente
            if dynamic_lock_counter > 0:
                dynamic_lock_counter -= 1
                # Continua a mostrare l'ultima dinamica
                final_char = last_dynamic_char
                print(f"🔒 Mantengo dinamica: {last_dynamic_char} ({dynamic_lock_counter} frame rimasti)")
            
            # 2. Se è una NUOVA lettera dinamica
            elif predicted_character in list(recognizer.dynamic_label_map.values()):
                last_dynamic_char = predicted_character
                dynamic_lock_counter = dynamic_lock_frames
                last_display_char = predicted_character
                display_counter = HOLD_FRAMES
                print(f"🎯 DINAMICA ({recognizer.model_type}): {predicted_character}")
                final_char = predicted_character
            
            # 3. Se è una lettera statica E non siamo in blocco
            elif predicted_character != last_display_char:
                last_display_char = predicted_character
                display_counter = HOLD_FRAMES
                print(f"📝 Nuova statica: {predicted_character}")
                final_char = predicted_character
            
            # 4. Altrimenti mantieni l'ultima visualizzata
            else:
                final_char = last_display_char
            
            # Visualizza
            if display_counter > 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, final_char, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 2, cv2.LINE_AA)
                display_counter -= 1
            # else: lettera sparisce


    cv2.imshow('Riconoscimento LIS', frame)
    cv2.waitKey(10) #Aspettiamo 25 millisecondi per ogni frame

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()