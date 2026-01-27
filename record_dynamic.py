import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from collections import deque

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Setup webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Directory per salvare i dati dinamici
DYNAMIC_DIR = './dynamic_data'
os.makedirs(DYNAMIC_DIR, exist_ok=True)

# Buffer per sequenza
sequence_buffer = deque(maxlen=30)
recording = False
current_gesture = ""
samples_recorded = 0
target_samples = 50  # Campioni da registrare per ogni gesto

print("=== STRUMENTO REGISTRAZIONE GESTI DINAMICI ===")
print("Comandi:")
print("  - 's' + lettera: inizia registrazione (es: 'sj' per lettera J)")
print("  - 'spazio': inizia/ferma acquisizione sequenza")
print("  - 'q': termina registrazione corrente")
print("  - 'x': esci dal programma")

# Chiedi quale lettera registrare
current_gesture = input("Inserisci lettera da registrare (es: J, Z): ").upper()
print(f"Registrerai la lettera: {current_gesture}")
print(f"Registra {target_samples} sequenze di 30 frame ciascuna")

sequence_data = []  # Qui salviamo tutte le sequenze

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # Disegna stato
    status_text = f"Gesto: {current_gesture} - Campioni: {samples_recorded}/{target_samples}"
    cv2.putText(frame, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if recording:
        cv2.putText(frame, "REGISTRANDO...", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Frame: {len(sequence_buffer)}/30", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Estrai landmarks
        first_hand = results.multi_hand_landmarks[0]
        data_aux = []
        
        for i in range(len(first_hand.landmark)):
            x = first_hand.landmark[i].x
            y = first_hand.landmark[i].y
            data_aux.append(x)
            data_aux.append(y)
        
        # Se stiamo registrando, aggiungi alla sequenza
        if recording and len(data_aux) == 42:
            sequence_buffer.append(data_aux)
            
            # Quando la sequenza è completa
            if len(sequence_buffer) == 30:
                # Salva la sequenza
                sequence_data.append(list(sequence_buffer.copy()))
                samples_recorded += 1
                print(f"Campione {samples_recorded} salvato!")
                
                # Resetta
                recording = False
                sequence_buffer.clear()
                
                # Controlla se abbiamo finito
                if samples_recorded >= target_samples:
                    print(f"Registrazione completata per '{current_gesture}'!")
                    break
    
    cv2.imshow('Registrazione Gesti Dinamici', frame)
    
    key = cv2.waitKey(10) & 0xFF
    
    if key == ord(' '):  # Spazio: inizia/ferma registrazione
        if not recording and samples_recorded < target_samples:
            recording = True
            sequence_buffer.clear()
            print("Inizio registrazione sequenza...")
        elif recording:
            recording = False
            print("Registrazione interrotta")
    
    elif key == ord('q'):  # Termina registrazione corrente
        if samples_recorded > 0:
            print(f"Registrazione interrotta. Campioni salvati: {samples_recorded}")
            break
        else:
            print("Nessun campione salvato")
    
    elif key == ord('x'):  # Esci
        print("Uscita...")
        break

# Salva i dati
if sequence_data:
    save_path = os.path.join(DYNAMIC_DIR, f"{current_gesture}_sequences.pickle")
    with open(save_path, 'wb') as f:
        pickle.dump({
            'gesture': current_gesture,
            'sequences': sequence_data,
            'samples': samples_recorded
        }, f)
    print(f"Dati salvati in: {save_path}")
    print(f"Sequenze salvate: {len(sequence_data)}")
    print(f"Frame per sequenza: 30")
    print(f"Features per frame: 42")

cap.release()
cv2.destroyAllWindows()