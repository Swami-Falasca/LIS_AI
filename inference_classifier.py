import cv2
import mediapipe as mp
import pickle
import numpy as np

#Carico il modello
model_dict = pickle.load(open('./model.p','rb'))
model = model_dict['model']

#se non va provare 1, 2...
#cap = cv2.VideoCapture(0)
#dovrebbe essere una nuova versione
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  

#Esattamente come in create dataset
#Detect and draw landmarks sulle immagini per la classificazione
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode= True, min_detection_confidence=0.3)



labels_dict = {0: 'a', 1: 'b', 2: 'c'}


while True:

    data_aux = []
    x_ = []
    y_ = []


    ret, frame = cap.read()

    #perché restituirebbe 3 valori (il terzo è canali)
    H, W, _ = frame.shape  

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Detect all the landmarks in this image
    results = hands.process(frame_rgb)

        #Itera nel risultato
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                    frame, #image to draw
                    hand_landmarks, #model output
                    mp_hands.HAND_CONNECTIONS, #hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
            )

        #Ora con i landmarks ottenuti andiamo a usare il modello
        for hand_landmarks in results.multi_hand_landmarks:
                
                 # Per OGNI mano, resetta data_aux
                data_aux = []


                #Creiamo array da landmarks
                for i in range(len(hand_landmarks.landmark)):
                    #x,y,z posizioni dei landmarks
                    #a noi servono solo x e y
                    #print(hand_landmarks.landmark[i])
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

                    x_.append(x)
                    y_.append(y)

        #questi sono i bordi del rettangolo che contengono la mano
        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)

        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        prediction = model.predict([np.asarray(data_aux)]) #[]sono per convertire l'input in lista, altrimenti dà errore

        #predicted_character = labels_dict[int(prediction[0])]
        predicted_character = str(prediction[0])  # Converti direttamente in stringa

        print(predicted_character)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 4)
        cv2.putText(frame, predicted_character, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0),3,cv2.LINE_AA)


    cv2.imshow('frame', frame)
    cv2.waitKey(10) #Aspettiamo 25 millisecondi per ogni frame

cap.release()
cv2.destroyAllWindows()