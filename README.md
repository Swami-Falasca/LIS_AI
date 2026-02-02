# LIS_AI

![image](https://github.com/user-attachments/assets/b407d6a6-0168-46d3-bbfd-cf7b587dc673)

📰 Descrizione
LIS_AI è un progetto dedicato alla creazione di un sistema di intelligenza artificiale che si basa sulla traduzione della lingua dei segni (LIS) alla lingua italiana. 
-	Il sistema si basa su un riconoscimento automatico per tradurre gesti in tempo reale
-	Supporta tutte le lettere dell’alfabeto 
-	Riconosce entrambe le mani

🎯 Obiettivi
-	Rilevare i segni LIS da immagini
-	Facilitare la comunicazione tra non udenti e udenti

⚙️ Setup
1.	Clona il repository: git clone https://github.com/Swami-Falasca/LIS_AI

2.	Installare le seguenti dipendenze: 
           - mediapipe 0.10.7
           - matplotlib 3.10.8
           - numpy 1.23.5
           - scikit-Learn 1.2.0
           - keras 2.13.1
           - tensorflow 2.13.0
    

3.	Avviare il backend
           - Eseguire "inference_server.py" nella cartella backend
  

4.	Avviare il frontend
           - "home.html" nella cartella frontend su localhost


Come allenare i modelli con il proprio dataset:
    
Lettere statiche:
1. Inserire il dataset etichettato nella cartella data
       
2. Eseguire il file "create_dataset.py" nella cartella dataset
       
3. Eseguire il file "train_static.py" nella cartella training per allenare il modello

Lettere dinamiche:
1. Registrare le lettere eseguendo il file "record_dynamic.py" nella cartella dataset

2. Eseguire il file "train_dynamic.py" nella cartella training per allenare il modello
