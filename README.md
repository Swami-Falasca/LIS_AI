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

2.	Crea un ambiente virtuale:

       python -m venv .venv

       .venv/bin/activate

3.	Installare le seguenti dipendenze: 
           - mediapipe 0.10.7
           - matplotlib 3.10.8
           - numpy 1.23.5
           - scikit-Learn 1.2.0
           - keras 2.13.1
           - tensorflow 2.13.0
    

4.	Avviare il backend
           - python inference_classifier
  

5.	Avviare il frontend
           - home.html su localhost


Come allenare i modelli con il proprio dataset:
    Lettere statiche:
           - Inserire il dataset etichettato nella cartella data
           - Eseguire il file "create_dataset.py" nella cartella dataset
           - Eseguire il file "train_static.py" nella cartella training per allenare il modello
    Lettere dinamiche
           - Registrare le lettere eseguendo il file "record_dynamic.py" nella cartella dataset
           - Eseguire il file "train_dynamic.py" nella cartella training per allenare il modello
