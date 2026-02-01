import cv2
import base64
import threading
import time
import json
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from inference_classifier import HybridGestureRecognizer, process_frame
import mediapipe as mp
import numpy as np

app = Flask(__name__)
CORS(app)  # Permette richieste dal frontend

# Inizializza MediaPipe una volta sola
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Inizializza riconoscitore
recognizer = HybridGestureRecognizer()

# Variabili per gestire lo stato
current_prediction = ""
prediction_history = []
is_processing = False

@app.route('/api/status', methods=['GET'])
def get_status():
    """Endpoint per verificare lo stato del server"""
    return jsonify({
        'status': 'online',
        'model_loaded': True,
        'dynamic_model': recognizer.has_dynamic_model,
        'message': 'Server LIS AI attivo e funzionante'
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint per riconoscimento da singolo frame"""
    global current_prediction, is_processing
    
    try:
        # Evita elaborazioni concorrenti
        if is_processing:
            return jsonify({
                'letter': current_prediction,
                'confidence': 0.95,
                'status': 'busy',
                'timestamp': time.time()
            }), 202
        
        is_processing = True
        
        # Ricevi frame base64
        data = request.json
        if 'frame' not in data:
            return jsonify({'error': 'No frame provided'}), 400
        
        # Decodifica base64
        frame_data = data['frame']
        if frame_data.startswith('data:image'):
            # Rimuovi header data:image/jpeg;base64,
            frame_data = frame_data.split(',')[1]
        
        nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Processa frame con la tua logica
        result = process_frame(frame, recognizer, hands)
        
        if result['has_hand']:
            current_prediction = result['letter']
            
            # Aggiungi alla cronologia
            prediction_record = {
                'letter': result['letter'],
                'timestamp': time.time(),
                'bbox': result['bbox']
            }
            prediction_history.append(prediction_record)
            
            # Mantieni solo ultimi 100 record
            if len(prediction_history) > 100:
                prediction_history.pop(0)
            
            response = {
                'letter': result['letter'],
                'confidence': 0.95,
                'has_hand': True,
                'bbox': result['bbox'], # [x_min, y_min, x_max, y_max]
                'landmarks': result.get('landmarks', []), # Assicurati che process_frame restituisca i punti!
                'timestamp': time.time()
            }
        else:
            response = {
                'letter': '',
                'confidence': 0.0,
                'has_hand': False,
                'message': 'Nessuna mano rilevata',
                'timestamp': time.time()
            }
        
        is_processing = False
        return jsonify(response)
        
    except Exception as e:
        is_processing = False
        print(f"Errore durante la predizione: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': time.time()
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Restituisce la cronologia delle predizioni"""
    return jsonify({
        'history': prediction_history[-20:],  # Ultimi 20 record
        'count': len(prediction_history),
        'current_prediction': current_prediction
    })

@app.route('/api/stream')
def stream():
    """Streaming Server-Sent Events per aggiornamenti in tempo reale"""
    def generate():
        last_prediction = ""
        while True:
            # Invia solo se c'è una nuova predizione
            if current_prediction and current_prediction != last_prediction:
                data = {
                    'letter': current_prediction,
                    'timestamp': time.time(),
                    'type': 'prediction'
                }
                yield f"data: {json.dumps(data)}\n\n"
                last_prediction = current_prediction
            
            # Invia anche heartbeat per mantenere connessione
            heartbeat = {
                'type': 'heartbeat',
                'timestamp': time.time()
            }
            yield f"data: {json.dumps(heartbeat)}\n\n"
            
            time.sleep(0.01) 
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/reset', methods=['POST'])
def reset():
    """Resetta lo stato del riconoscitore"""
    global current_prediction, prediction_history
    current_prediction = ""
    prediction_history = []
    
    # Reset del buffer sequenze nel riconoscitore
    recognizer.sequence_buffer.clear()
    
    return jsonify({
        'status': 'reset',
        'message': 'Stato del riconoscitore resettato',
        'timestamp': time.time()
    })

@app.route('/api/predict_landmarks', methods=['POST'])
def predict_landmarks():
    data = request.json
    landmarks = data.get('landmarks') # Riceve array di 42 numeri
    
    if not landmarks or len(landmarks) != 42:
        return jsonify({'letter': '?'}), 400

    # Usa il tuo recognizer esistente
    prediction = recognizer.predict(np.array(landmarks))
    return jsonify({'letter': prediction})

if __name__ == '__main__':
    print("=" * 50)
    print("Server LIS AI in avvio...")
    print(f"Modello statico caricato")
    if recognizer.has_dynamic_model:
        print(f"Modello dinamico caricato: {list(recognizer.dynamic_label_map.values())}")
    print("Endpoint disponibili:")
    print("  GET  /api/status     - Stato del server")
    print("  POST /api/predict    - Riconoscimento da frame")
    print("  GET  /api/history    - Cronologia predizioni")
    print("  GET  /api/stream     - Streaming in tempo reale (SSE)")
    print("  POST /api/reset      - Reset stato riconoscitore")
    print("=" * 50)
    
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)