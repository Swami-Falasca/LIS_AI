import pickle
import numpy as np
import os
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Directory dati
DYNAMIC_DIR = '../dynamic_data'
MODELS_DIR = '../models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Funzioni di supporto
def flip_hand_sequence(sequence):
    """Ribalta sequenza landmarks"""
    flipped = sequence.copy()
    flipped[:, 0::2] = 1.0 - flipped[:, 0::2]
    return flipped

def preprocess_sequence(sequence):
    """Applica filtri per ridurre rumore"""
    seq_array = np.array(sequence)
    
    # 1. Normalizzazione per gesto (non per frame)
    seq_array = (seq_array - np.min(seq_array)) / (np.max(seq_array) - np.min(seq_array) + 1e-8)
    
    # 2. Smoothing (media mobile)
    smoothed = np.zeros_like(seq_array)
    for i in range(1, len(seq_array)-1):
        smoothed[i] = 0.25*seq_array[i-1] + 0.5*seq_array[i] + 0.25*seq_array[i+1]
    
    return smoothed


def load_dynamic_data():
    """Carica dati per tutti i modelli"""
    sequences = []
    labels = []
    label_map = {}
    
    data_files = [f for f in os.listdir(DYNAMIC_DIR) if f.endswith('_sequences.pickle')]
    
    if not data_files:
        print("Nessun dato dinamico trovato!")
        return None, None, None
    
    print(f"Trovati {len(data_files)} file di dati dinamici:")
    
    for idx, filename in enumerate(data_files):
        filepath = os.path.join(DYNAMIC_DIR, filename)
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        gesture_name = data['gesture']
        gesture_sequences = data['sequences']
        
        print(f"  - {gesture_name}: {len(gesture_sequences)} sequenze")
        
        # Processa ogni sequenza
        for seq in gesture_sequences:
            # APPLICA PREPROCESSING QUI
            processed_seq = preprocess_sequence(seq)
            sequences.append(processed_seq)
            labels.append(idx)
        
        # Flippate (data augmentation)
        for seq in gesture_sequences:
            flipped_seq = flip_hand_sequence(np.array(seq))
            # Preprocess anche le sequenze flippate
            processed_flipped = preprocess_sequence(flipped_seq)
            sequences.append(processed_flipped)
            labels.append(idx)
        
        label_map[idx] = gesture_name
    
    print(f"\nTotale sequenze: {len(sequences)} (originali + flipped)")
    return np.array(sequences), np.array(labels), label_map

# ==================== CLASSIFICATORI ====================
def train_lstm(X_train, X_test, y_train, y_test, label_map):
    """LSTM Neural Network"""
    try:
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        from keras.utils import to_categorical
    except ImportError:
        print("  Keras/TensorFlow non installato. Salto...")
        return None, None, None
    
    print("  Training LSTM...")
    start_time = time.time()
    
    # One-hot encoding
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    
    # Modello
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(len(label_map), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Addestra
    history = model.fit(
        X_train, y_train_cat,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )
    
    train_time = time.time() - start_time
    
    # Valuta
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Metriche
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results = {
        'model_name': 'LSTM',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'train_time': train_time,
        'epochs': 50,
        'input_shape': (X_train.shape[1], X_train.shape[2])
    }
    
    return model, results, (X_train.shape[1], X_train.shape[2])

def train_gru(X_train, X_test, y_train, y_test, label_map):
    """GRU Neural Network"""
    try:
        from keras.models import Sequential
        from keras.layers import GRU, Dense, Dropout
        from keras.utils import to_categorical
    except ImportError:
        print("  Keras/TensorFlow non installato. Salto...")
        return None, None, None
    
    print("  Training GRU...")
    start_time = time.time()
    
    # One-hot encoding
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    
    # Modello
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        GRU(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(len(label_map), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Addestra
    history = model.fit(
        X_train, y_train_cat,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )
    
    train_time = time.time() - start_time
    
    # Valuta
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Metriche
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results = {
        'model_name': 'GRU',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'train_time': train_time,
        'epochs': 50,
        'input_shape': (X_train.shape[1], X_train.shape[2])
    }
    
    return model, results, (X_train.shape[1], X_train.shape[2])


# ==================== MAIN ====================

def main():
    print("="*60)
    print("TEST COMPARATIVO CLASSIFICATORI PER GESTI DINAMICI")
    print("="*60)
    
    # Carica dati
    X, y, label_map = load_dynamic_data()
    if X is None:
        return
    
    print(f"\n📊 DATASET:")
    print(f"  Sequenze totali: {X.shape[0]}")
    print(f"  Shape sequenza: {X.shape[1:]} (30 frame × 42 features)")
    print(f"  Classi: {len(label_map)}")
    for idx, gesture in label_map.items():
        print(f"    {idx}: {gesture}")
    
    # Split dati
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n🔀 SPLIT DATASET:")
    print(f"  Train: {X_train.shape[0]} sequenze")
    print(f"  Test:  {X_test.shape[0]} sequenze")
    
    # Lista classificatori da testare
    classifiers = [
        ('LSTM', train_lstm),
        ('GRU', train_gru)
    ]
    
    all_results = {}
    models_info = {}
    
    print(f"\nINIZIO TEST CLASSIFICATORI")
    print("="*60)
    
    for clf_name, clf_func in classifiers:
        print(f"\n▶Testing: {clf_name}")
        print("-"*40)
        
        try:
            model, results, input_shape = clf_func(X_train, X_test, y_train, y_test, label_map)
            
            if model is not None and results is not None:
                all_results[clf_name] = results
                
                # Salva modello
                model_filename = f"{MODELS_DIR}/{clf_name.lower()}_model.p"
                with open(model_filename, 'wb') as f:
                    pickle.dump({
                        'model': model,
                        'label_map': label_map,
                        'input_shape': input_shape,
                        'model_name': clf_name,
                        'metrics': results
                    }, f)
                
                models_info[clf_name] = {
                    'file': model_filename,
                    'input_shape': input_shape
                }
                
                print(f"Salvato: {model_filename}")
                print(f"Accuracy: {results['accuracy']:.2%}")
                print(f"Tempo training: {results['train_time']:.1f}s")
            else:
                print(f"{clf_name} non disponibile o errore")
                
        except Exception as e:
            print(f"Errore {clf_name}: {e}")
    
    # Risultati comparativi
    print(f"\n" + "="*60)
    print("RISULTATI COMPARATIVI")
    print("="*60)
    
    if all_results:
        # Ordina per accuracy
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        print(f"\n{'Classificatore':<15} {'Accuracy':<10} {'F1-Score':<10} {'Tempo (s)':<10}")
        print("-"*45)
        
        for clf_name, metrics in sorted_results:
            print(f"{clf_name:<15} {metrics['accuracy']:<10.2%} {metrics['f1_score']:<10.2%} {metrics['train_time']:<10.1f}")
        
        # Salva report JSON
        report_file = f"{MODELS_DIR}/comparison_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'dataset_info': {
                    'total_sequences': X.shape[0],
                    'sequence_shape': X.shape[1:],
                    'num_classes': len(label_map),
                    'classes': label_map
                },
                'results': all_results,
                'models_info': models_info,
                'best_model': sorted_results[0][0] if sorted_results else None
            }, f, indent=2)
        
        print(f"\n📄 Report salvato: {report_file}")
        
        # Miglior modello
        best_clf = sorted_results[0][0]
        print(f"\n🏆 MIGLIOR CLASSIFICATORE: {best_clf}")
        print(f"   Accuracy: {all_results[best_clf]['accuracy']:.2%}")
        print(f"   File modello: {models_info[best_clf]['file']}")
        
        # Crea link simbolico al miglior modello
        try:
            best_model_file = models_info[best_clf]['file']
            os.system(f'copy "{best_model_file}" "{MODELS_DIR}/best_model.p"')
            print(f"   ✅ Miglior modello copiato come: {MODELS_DIR}/best_model.p")
        except:
            pass
    
    else:
        print("Nessun modello addestrato con successo")
    
    print(f"\nTutti i modelli salvati in: {MODELS_DIR}/")

if __name__ == "__main__":
    main()