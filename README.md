# LIS_AI
<div align="center">
<img width="300" alt="LIS_AI" src="https://github.com/user-attachments/assets/b407d6a6-0168-46d3-bbfd-cf7b587dc673" \>
</div>

# Il progetto
**LIS_AI** è un progetto dedicato alla creazione di un sistema di intelligenza artificiale che si basa sulla traduzione della lingua dei segni (LIS) alla lingua italiana. 
-	Il sistema si basa su un riconoscimento automatico per tradurre gesti in tempo reale
-	Supporta tutte le lettere dell’alfabeto 
-	Riconosce entrambe le mani

# Obiettivi
-	Rilevare i segni LIS da immagini
-	Facilitare la comunicazione tra non udenti e udenti

# Setup
1.	Clonare il repository:
   
  	```bash
           git clone https://github.com/Swami-Falasca/LIS_AI

2.	Installare le dipendenze dal file `requirements.txt`:

  	```bash
           pip install requirements.txt

3.	Avviare il backend
           - Eseguire `inference_server.py` nella cartella backend
  

4.	Avviare il frontend
           - `home.html` nella cartella frontend su localhost


# Come allenare i modelli con il proprio dataset
    
Lettere statiche:
1. Inserire il dataset etichettato nella cartella `data`
       
2. Eseguire il file `create_dataset.py` nella cartella dataset
       
3. Eseguire il file `train_static.py` nella cartella training per allenare il modello

Lettere dinamiche:
1. Registrare le lettere eseguendo il file `record_dynamic.py` nella cartella dataset

2. Eseguire il file `train_dynamic.py` nella cartella training per allenare il modello

# Tecnologie Utilizzate
<div align="left">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
<img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
<img src="https://img.shields.io/badge/Matplotlib-71D291?logo=matplotlib&style=for-the-badge&logoColor=fff" alt="MatPlotLib" />
<img src="https://img.shields.io/badge/Keras-D00000?logo=keras&style=for-the-badge&logoColor=fff" alt="Keras" />
</div>
