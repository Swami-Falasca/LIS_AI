import pickle
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Directory dati dinamici
DYNAMIC_DIR = './dynamic_data'

#Ribalta i landmarks
def flip_hand_sequence(sequence):
    """Ribalta sequenza usando numpy vettorizzato"""
    # sequence shape: (30, 42)
    flipped = sequence.copy()
    
    # Ribalta coordinate X (indici pari: 0, 2, 4, ...)
    flipped[:, 0::2] = 1.0 - flipped[:, 0::2]
    
    return flipped

def load_dynamic_data():
    """Carica tutti i dati dinamici registrati"""
    sequences = []
    labels = []
    label_map = {}
    
    # Trova tutti i file .pickle nella directory
    data_files = [f for f in os.listdir(DYNAMIC_DIR) if f.endswith('_sequences.pickle')]
    
    if not data_files:
        print("Nessun dato dinamico trovato!")
        print(f"Assicurati di avere file in: {DYNAMIC_DIR}")
        print("Usa prima record_dynamic.py per registrare gesti")
        return None, None, None
    
    print(f"Trovati {len(data_files)} file di dati dinamici:")
    
    for idx, filename in enumerate(data_files):
        filepath = os.path.join(DYNAMIC_DIR, filename)
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        gesture_name = data['gesture']
        gesture_sequences = data['sequences']
        
        print(f"  - {gesture_name}: {len(gesture_sequences)} sequenze")
        
        # Aggiungi alle liste
        sequences.extend(gesture_sequences)
        labels.extend([idx] * len(gesture_sequences))

        for seq in gesture_sequences:
            sequences.append(seq)
            labels.append(idx)
            
            # Aggiungi sequenze FLIPPATE (data augmentation)
            flipped_seq = flip_hand_sequence(np.array(seq))
            sequences.append(flipped_seq)
            labels.append(idx)

        label_map[idx] = gesture_name

    print(f"\nData augmentation: {len(sequences)} sequenze totali (originali + flipped)")
    return np.array(sequences), np.array(labels), label_map

def build_lstm_model(input_shape, num_classes):
    """Costruisci modello LSTM per sequenze temporali"""
    model = Sequential([
        # Primo layer LSTM
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        
        # Secondo layer LSTM
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        
        # Terzo layer LSTM
        LSTM(64),
        Dropout(0.3),
        
        # Layer denso
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("=== ADDESTRAMENTO MODELLO DINAMICO LSTM ===")
    
    # Carica dati
    X, y, label_map = load_dynamic_data()
    
    if X is None:
        return
    
    print(f"\nShape dati:")
    print(f"  X (sequenze): {X.shape}")
    print(f"  y (labels): {y.shape}")
    print(f"  Classi: {len(label_map)}")
    
    for idx, gesture in label_map.items():
        print(f"    {idx}: {gesture}")
    
    # Preparazione dati
    # X shape: (num_sequenze, 30, 42)
    # y shape: (num_sequenze,)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nSplit dataset:")
    print(f"  Train: {X_train.shape[0]} sequenze")
    print(f"  Test:  {X_test.shape[0]} sequenze")
    
    # Costruisci modello
    input_shape = (X_train.shape[1], X_train.shape[2])  # (30, 42)
    num_classes = len(label_map)
    
    model = build_lstm_model(input_shape, num_classes)
    
    print("\nModello LSTM creato:")
    model.summary()
    
    # Addestra
    print("\nInizio addestramento...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    # Valuta
    print("\nValutazione sul test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Accuracy test: {test_acc:.2%}")
    print(f"  Loss test: {test_loss:.4f}")
    
    # Salva modello
    model.save('dynamic_gesture_model.h5')
    print("\nModello salvato come: dynamic_gesture_model.h5")
    
    # Salva anche con pickle per compatibilità
    with open('dynamic_model.p', 'wb') as f:
        pickle.dump({
            'model': model,
            'label_map': label_map,
            'input_shape': input_shape,
            'accuracy': test_acc
        }, f)
    
    print("Modello salvato come: dynamic_model.p")
    
    # Test predizioni
    print("\nTest predizioni su alcune sequenze:")
    for i in range(min(3, len(X_test))):
        sample = X_test[i:i+1]
        true_label = y_test[i]
        pred = model.predict(sample, verbose=0)
        pred_label = np.argmax(pred[0])
        
        print(f"  Sequenza {i+1}:")
        print(f"    Vera lettera: {label_map[true_label]}")
        print(f"    Predetta: {label_map.get(pred_label, 'Unknown')}")
        print(f"    Confidence: {np.max(pred[0]):.2%}")

if __name__ == "__main__":
    main()