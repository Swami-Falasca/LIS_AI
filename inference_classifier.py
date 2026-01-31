import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque

#Esattamente come in create dataset
#Detect and draw landmarks sulle immagini per la classificazione
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#DINAMICO
sequence_buffer = deque(maxlen=30)  # Buffer per 30 frame
dynamic_mode = False  # True per attivare riconoscimento dinamico

#Riconosce se c'è movimento nella sequenza
def analyze_movement(sequence_buffer):
    if len(sequence_buffer) < 2:
        return False
    
    #converte in array numpy
    sequence_array = np.array(list(sequence_buffer))
    
    #calcola varianza delle posizioni nel tempo
    #(se le dita si muovono, la varianza sarà alta)
    position_variance = np.var(sequence_array, axis=0)
    
    #prendi solo le coordinate X indici pari
    x_variance = position_variance[::2]
    
    #se la varianza media supera una soglia, c'è movimento
    movement_threshold = 0.001  #Soglia empirica
    mean_variance = np.mean(x_variance)
    
    return mean_variance > movement_threshold

class HybridGestureRecognizer:
    def __init__(self):
        # Carica modello statico
        self.static_model = pickle.load(open('./model.p','rb'))['model']
        
        # Prova a caricare modello dinamico (se esiste)
        try:
            with open('./dynamic_models/lstm_model.p', 'rb') as f:
                dynamic_data = pickle.load(f)
                self.dynamic_model = dynamic_data['model']
                self.dynamic_label_map = dynamic_data['label_map']
                self.has_dynamic_model = True
                print(f"Modello dinamico caricato. Lettere: {list(self.dynamic_label_map.values())}")
        except:
            self.has_dynamic_model = False
            print("Modello dinamico non trovato. Userò solo modello statico.")
        
        # Buffer per sequenze
        self.sequence_buffer = deque(maxlen=30)
        self.current_prediction = ""

        # Lettere che potrebbero essere confuse statiche/dinamiche
        #self.ambiguous_letters = ['i', 'h', 'n', 'q','o','t','m','x']  # 'I' potrebbe essere 'J', ecc.
    
    def predict(self, landmarks):
        """Predice usando entrambi i modelli automaticamente"""
        
        # 1. Prova modello statico
        static_pred = self.static_model.predict([landmarks])[0]
        
        # 2. Se è una lettera che potrebbe essere dinamica...
        if self.has_dynamic_model:
            # Aggiungi alla sequenza
            self.sequence_buffer.append(landmarks)
            
            # Quando abbiamo abbastanza frame, analizza
            if len(self.sequence_buffer) == 30:
                # PRIMA di usare modello dinamico, controlla se c'è MOVIMENTO
                has_movement = analyze_movement(self.sequence_buffer)  # <-- AGGIUNTA
                
                if has_movement:  # <-- SOLO SE C'È MOVIMENTO
                    #Usa modello dinamico
                    sequence_array = np.array(self.sequence_buffer).reshape(1, 30, -1)
                    
                    #Predici con modello dinamico
                    dynamic_pred_proba = self.dynamic_model.predict(sequence_array, verbose=0)
                    dynamic_pred_idx = np.argmax(dynamic_pred_proba[0])
                    dynamic_pred = self.dynamic_label_map[dynamic_pred_idx]
                    
                    #Controlla confidence
                    confidence = np.max(dynamic_pred_proba[0])
                    
                    #Se confidence alta (>80%) e diversa da statica, usa dinamica
                    if confidence > 0.8 and dynamic_pred != static_pred:
                        self.current_prediction = dynamic_pred
                        print(f"[Dinamico] Riconosciuto: {dynamic_pred} (confidence: {confidence:.1%})")
                    else:
                        self.current_prediction = static_pred
                        print(f"[Statico] Nessun movimento significativo")
                else:
                    # Nessun movimento, mantieni statico
                    self.current_prediction = static_pred
                    print(f"[Statico] Nessun movimento rilevato")
                
                #reset buffer
                self.sequence_buffer.clear()
            
            else:
                #Sequenza non completa, usa statica
                self.current_prediction = static_pred
                return static_pred

        else:
            #Lettera non ambigua o senza modello dinamico
            self.current_prediction = static_pred
            self.sequence_buffer.clear()  #Resetta se stava accumulando
            return static_pred

        return self.current_prediction

#Pocessa frame senza avviare il ciclo while
def process_frame(frame, recognizer, hands):
    """Processa un singolo frame e restituisce la predizione"""
    H, W, _ = frame.shape
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #Detect all the landmarks in this image
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        #Solo la prima mano per la predizione
        first_hand = results.multi_hand_landmarks[0]
        data_aux = []
        x_ = []
        y_ = []
        
        #Creiamo array da landmarks della PRIMA mano
        for i in range(len(first_hand.landmark)):
            x = first_hand.landmark[i].x
            y = first_hand.landmark[i].y
            data_aux.append(x)
            data_aux.append(y)
            x_.append(x)
            y_.append(y) 
        
        #Usa il riconoscitore ibrido
        if len(data_aux) == 42:
            predicted_character = recognizer.predict(np.asarray(data_aux))
            
            #questi sono i bordi del rettangolo che contengono la mano
            x1 = int(min(x_) * W) - 10  # -10 per margine
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10
            
            landmarks_pixels = []
            for i in range(0, len(data_aux), 2):
                landmarks_pixels.append({
                    'x': int(data_aux[i] * W),
                    'y': int(data_aux[i+1] * H)
                })

            return {
            'letter': predicted_character,
            'bbox': [x1, y1, x2, y2], # Usiamo una lista [] invece di tupla () per JSON
            'landmarks': landmarks_pixels, # Inviamo la lista di oggetti x,y
            'has_hand': True
        }
    
    return {
            'letter': predicted_character,
            'bbox': [x1, y1, x2, y2], # Usiamo una lista [] invece di tupla () per JSON
            'landmarks': landmarks_pixels, # Inviamo la lista di oggetti x,y
            'has_hand': True
        }

def main():
    """Funzione principale per eseguire il riconoscimento standalone"""
    #se non va provare 1, 2...
    #cap = cv2.VideoCapture(0)
    #dovrebbe essere una nuova versione
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # static_image_mode=False per video, True solo per immagini statiche
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
    
    #Carico il modello
    recognizer = HybridGestureRecognizer()
    
    last_display_char = ""
    display_counter = 0
    HOLD_FRAMES = 40  # Mostra per 40 frame
    
    last_dynamic_char = ""
    dynamic_lock_frames = 60  # Blocca per 60 frame dopo una dinamica
    dynamic_lock_counter = 0

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
                elif predicted_character in ['S', 'Z', 'J', 'SG']:  # O lista tue dinamiche
                    last_dynamic_char = predicted_character
                    dynamic_lock_counter = dynamic_lock_frames
                    last_display_char = predicted_character
                    display_counter = HOLD_FRAMES
                    print(f"🎯 NUOVA DINAMICA: {predicted_character} (blocco {dynamic_lock_frames} frame)")
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

# Solo se eseguito direttamente, avvia il main
if __name__ == "__main__":
    main()