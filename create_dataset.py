import os
import pickle #salvare dataset ecc
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

#py -3.10 -m pip install mediapipe==0.10.7
#PER INSTALLARE CORRETTA VERSIONE MEDIAPIPE CON PYTHON 3.10

#Per cambiare versione di python: ctrl+shift+p, select interpreter

#cd "C:\Users\isabe\OneDrive\Desktop\FIA" per direcotry corretta

def normalize_landmarks(landmarks):
    """Normalizza landmarks rispetto al polso (punto 0)"""

    if len(landmarks) == 0:
        return landmarks  # Restituisce lista vuota se nessun landmark

    wrist_x = landmarks[0]  # x polso
    wrist_y = landmarks[1]  # y polso
    
    normalized = []
    for i in range(0, len(landmarks), 2):
        # Normalizza rispetto al polso
        norm_x = landmarks[i] - wrist_x
        norm_y = landmarks[i+1] - wrist_y
        normalized.extend([norm_x, norm_y])
    
    return normalized


#Detect and draw landmarks sulle immagini per la classificazione
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode= True, min_detection_confidence=0.3)

#Directory con il dataset
DATA_DIR = './data'

data = [] #dati che produrremo
labels = [] #categorie per ogni dato

for dir_ in os.listdir(DATA_DIR): #Itero nella directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Detect all the landmarks in this image
        results = hands.process(img_rgb)

        #Itera nel risultato
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                #Creiamo array da landmarks
                for i in range(len(hand_landmarks.landmark)):
                    #x,y,z posizioni dei landmarks
                    #a noi servono solo x e y
                    #print(hand_landmarks.landmark[i])
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)


            if data_aux:  # Solo se abbiamo landmarks
                # Normalizzazione dei landmarks
                normalized_data = normalize_landmarks(data_aux)

                #così facendo creo il dataset
                data.append(normalized_data) #salvo tutti gli array con le coordinate nell'array di dati
                labels.append(dir_)

                # Aggiungi dati FLIPPATI (data augmentation)
                flipped_aux = []
                for i in range(0, len(data_aux), 2):
                    x = 1.0 - data_aux[i]  # Ribalta X
                    y = data_aux[i + 1]
                    flipped_aux.extend([x, y])
            
                # Normalizza anche i dati flippati
                normalized_flipped = normalize_landmarks(flipped_aux)
                data.append(normalized_flipped)
                labels.append(dir_)

                #Landmarks (dimostrazione)
            '''
                per ogni risultato disegnamo i landmarks
                mp_drawing.draw_landmarks(
                    img_rgb, #image to draw
                    hand_landmarks, #model output
                    mp_hands.HAND_CONNECTIONS, #hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
 
                '''
        else:
            print(f"⚠️ Nessuna mano rilevata in {dir_}/{img_path}")
'''
Per mostrare le immagini
        plt.figure()
        plt.imshow(img_rgb)

plt.show()
'''

#Costruiremo un classificatore con queste informazioni
print(f"Dataset creato: {len(data)} campioni, {len(set(labels))} classi")
print(f"Classi: {set(labels)}")

f = open('data.pickle', 'wb') #da libreria pickle
pickle.dump({'data':data, 'labels':labels}, f)
f.close()

print("Dataset salvato in data.pickle")


