const BACKEND_URL = 'http://127.0.0.1:5000';
let mediaPipeCamera = null;

const elements = {
    webcam: document.getElementById('webcam'),
    outputCanvas: document.getElementById('output-canvas'),
    btnStart: document.getElementById('btn-start'),
    btnStop: document.getElementById('btn-stop'),
    resultDisplay: document.getElementById('result-display')
};

let appState = {
    isRunning: false
};

// Configura MediaPipe
const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

hands.onResults(onResults);

function onResults(results) {
    
    elements.outputCanvas.style.display = 'block'; // Forza la visibilità

    // Forza dimensioni canvas uguali al video
    elements.outputCanvas.width = 640;
    elements.outputCanvas.height = 480;
    
    const ctx = elements.outputCanvas.getContext('2d');
    ctx.clearRect(0, 0, 640, 480);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];
        
        // DISEGNA SUBITO (così vedi se funziona)
        drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 5});
        drawLandmarks(ctx, landmarks, {color: '#FF0000', lineWidth: 2});

        // CHIAMA IL BACKEND
        const flatLandmarks = landmarks.flatMap(l => [l.x, l.y]);
        sendToBackend(flatLandmarks);
    }
}

async function sendToBackend(landmarks) {
    try {
        const response = await fetch(`${BACKEND_URL}/api/predict_landmarks`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ landmarks: landmarks })
        });
        const data = await response.json();
       if (data.letter) {
    // 1. FORZA MAIUSCOLO
    const upperLetter = data.letter.toUpperCase();
    
    // 2. IMPOSTA NEL DISPLAY
    elements.resultDisplay.textContent = upperLetter;
       }
    } catch (e) {
        console.warn("Server offline o rotta mancante");
    }
}

async function startRecognition() {
    try {
        appState.isRunning = true;
        elements.btnStart.disabled = true;
        
        // MOSTRA IL VIDEO E NASCONDI IL POSTO VUOTO
        elements.webcam.style.display = 'block';
        if (elements.videoPlaceholder) {
            elements.videoPlaceholder.style.display = 'none';
        }

        if (!mediaPipeCamera) {
            mediaPipeCamera = new Camera(elements.webcam, {
                onFrame: async () => {
                    await hands.send({image: elements.webcam});
                },
                width: 640,
                height: 480
            });
        }
        await mediaPipeCamera.start();
        console.log("Webcam attiva!");
    } catch (err) {
        console.error("Errore webcam:", err);
    }
}

function stopRecognition() {
    appState.isRunning = false;
    
    if (mediaPipeCamera) {
        mediaPipeCamera.stop();
    }

    // 1. FERMA DAVVERO IL FLUSSO VIDEO (Rimuove il "fermo immagine")
    if (elements.webcam.srcObject) {
        const tracks = elements.webcam.srcObject.getTracks();
        tracks.forEach(track => track.stop()); // Spegne fisicamente la cam
        elements.webcam.srcObject = null;      // Svuota il tag video
    }

    // 2. Pulisci il Canvas
    const ctx = elements.outputCanvas.getContext('2d');
    ctx.clearRect(0, 0, elements.outputCanvas.width, elements.outputCanvas.height);
    
    // 3. Nascondi gli elementi e mostra il placeholder
    elements.webcam.style.display = 'none';
    elements.outputCanvas.style.display = 'none';
    
    if (elements.videoPlaceholder) {
        elements.videoPlaceholder.style.display = 'flex';
    }

    // 4. Ripristino interfaccia
    elements.btnStart.disabled = false;
    elements.resultDisplay.textContent = "-";
    
    console.log("Webcam spenta e stream rimosso.");
}

elements.btnStart.onclick = startRecognition;
elements.btnStop.onclick = stopRecognition;