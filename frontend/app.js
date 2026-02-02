const BACKEND_URL = 'http://127.0.0.1:5000';
let mediaPipeCamera = null;

let sessionTimer = null;
let secondsElapsed = 0;

const elements = {
    // Pulsanti
    btnStart: document.getElementById('btn-start'),
    btnStop: document.getElementById('btn-stop'),
    btnSave: document.getElementById('btn-test'), // ID nel tuo HTML è btn-test
    btnClear: document.getElementById('btn-clear'),
    btnSpace: document.getElementById('btn-space'),
    
    // Video e Canvas
    webcam: document.getElementById('webcam'),
    outputCanvas: document.getElementById('output-canvas'),
    videoPlaceholder: document.getElementById('video-placeholder'),
    
    // Display Risultati
    resultDisplay: document.getElementById('detected-letter'), // La lettera gigante
    resultActive: document.getElementById('result-active'),
    resultPlaceholder: document.querySelector('.result-placeholder'),
    confidenceFill: document.getElementById('confidence-fill'),
    confidenceValue: document.getElementById('confidence-value'),
    
    // Cronologia
    historyList: document.getElementById('history-list'),
    historyPlaceholder: document.getElementById('history-placeholder')
};

let appState = {
    isRunning: false,
    currentSentence: "" // Per memorizzare la frase
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

// Cambia il testo del pulsante appena carica la pagina
elements.btnSave.innerHTML = '<i class="fas fa-save"></i> Salva';

hands.onResults(onResults);

function onResults(results) {
    
    elements.outputCanvas.style.display = 'block'; // Forza la visibilità

    // Forza dimensioni canvas uguali al video
    elements.outputCanvas.width = 640;
    elements.outputCanvas.height = 480;
    
    const ctx = elements.outputCanvas.getContext('2d');
    ctx.clearRect(0, 0, 640, 480);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {

        // Mostra il layout della lettera, nascondi "Avvia riconoscimento"
        elements.resultActive.style.display = 'block';
        elements.resultPlaceholder.style.display = 'none';

        const landmarks = results.multiHandLandmarks[0];
        
        ctx.shadowBlur = 10;
        drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 5});
        drawLandmarks(ctx, landmarks, {color: '#FF0000', lineWidth: 2});
        ctx.shadowBlur = 0;

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
            const letter = data.letter.toUpperCase();
            elements.resultDisplay.textContent = letter;
            
            // Gestione barra confidenza (se il server non la manda, usiamo 95%)
            const conf = data.confidence || 0;
            elements.confidenceFill.style.width = conf + "%";
            elements.confidenceValue.textContent = conf + "%";
            elements.confidenceFill.style.background = conf > 80 ? '#10b981' : '#f59e0b';
       }
    } catch (e) {
        console.warn("Server offline o rotta mancante");
    }
}

function startTimer() {
    if (sessionTimer) clearInterval(sessionTimer);
    secondsElapsed = 0;
    const timerElement = document.getElementById('session-time');
    sessionTimer = setInterval(() => {
        secondsElapsed++;
        const mins = Math.floor(secondsElapsed / 60).toString().padStart(2, '0');
        const secs = (secondsElapsed % 60).toString().padStart(2, '0');
        if (timerElement) timerElement.textContent = `${mins}:${secs}`;
    }, 1000);
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
    
    // Ferma il timer
    if (sessionTimer) {
        clearInterval(sessionTimer);
        // Lasciamo il tempo finale visibile o resettiamo a 00:00
    }

    if (mediaPipeCamera) {
        mediaPipeCamera.stop();
    }

    // Spegne fisicamente la webcam
    if (elements.webcam.srcObject) {
        const tracks = elements.webcam.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        elements.webcam.srcObject = null;
    }

    // Pulisci il Canvas
    const ctx = elements.outputCanvas.getContext('2d');
    ctx.clearRect(0, 0, elements.outputCanvas.width, elements.outputCanvas.height);
    
    // --- FIX PUNTO 4: CHIUDI RIQUADRO LETTERA ---
    elements.resultActive.style.display = 'none'; 
    elements.resultPlaceholder.style.display = 'flex';
    elements.resultDisplay.textContent = "-";

    // Ripristina placeholder video
    elements.webcam.style.display = 'none';
    elements.outputCanvas.style.display = 'none';
    if (elements.videoPlaceholder) {
        elements.videoPlaceholder.style.display = 'flex';
        // --- FIX PUNTO 5: AGGIORNA TESTO WEBCAM ---
        elements.videoPlaceholder.querySelector('p').textContent = "Webcam in attesa di avvio";
    }

    elements.btnStart.disabled = false;
    elements.btnStart.innerHTML = '<i class="fas fa-play"></i> Avvia';
}

// Funzione per aggiornare la visualizzazione della frase (Cronologia)
function updateHistoryUI() {
    if (appState.currentSentence.length > 0) {
        elements.historyPlaceholder.style.display = 'none';
        elements.historyList.style.display = 'block'; // Usiamo block per il testo continuo
        
        // Non creiamo più tanti div separati, ma un unico blocco di testo
        // con uno span per gestire lo stile del cursore o del testo
        elements.historyList.innerHTML = `
            <div class="sentence-text">${appState.currentSentence}</div>
        `;
    } else {
        elements.historyPlaceholder.style.display = 'flex';
        elements.historyList.style.display = 'none';
    }
}

const statusDot = document.querySelector('.status-dot');
const statusText = document.querySelector('.status-text');

async function updateSystemStatus() {
    const isOnline = await checkServerStatus();
    if (isOnline) {
        statusDot.style.background = '#10b981'; // Verde
        statusDot.style.boxShadow = '0 0 10px #10b981';
        statusText.textContent = 'Sistema Online';
        statusText.style.color = '#10b981';
    } else {
        statusDot.style.background = '#ef4444'; // Rosso
        statusDot.style.boxShadow = '0 0 10px #ef4444';
        statusText.textContent = 'Server Disconnesso';
        statusText.style.color = '#ef4444';
    }
}

// Controlla ogni 5 secondi
setInterval(updateSystemStatus, 5000);
updateSystemStatus(); // Esegui subito al caricamento

async function checkServerStatus() {
    try {
        const res = await fetch(`${BACKEND_URL}/api/status`);
        return res.ok;
    } catch (e) { return false; }
}

elements.btnStart.onclick = async () => {
    elements.btnStart.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Connessione...';
    
    const isOnline = await checkServerStatus();
    
    if (!isOnline) {
        alert("Errore: Il server Python non risponde. Assicurati che inference_server.py sia attivo.");
        elements.btnStart.innerHTML = '<i class="fas fa-play"></i> Avvia';
        return;
    }
    
    if (elements.videoPlaceholder) elements.videoPlaceholder.style.setProperty('display', 'none', 'important');
    startTimer();

    elements.webcam.width = 640;
    elements.webcam.height = 480;

    // AGGIUNTE LE PARENTESI () per chiamare la funzione
    await startRecognition(); 
    elements.btnStart.innerHTML = '<i class="fas fa-check"></i> Connesso';
};

elements.btnStop.onclick = stopRecognition;

// Evento Salva Lettera
elements.btnSave.onclick = () => {
    const letter = elements.resultDisplay.textContent;
    if (letter && letter !== "-") {
        appState.currentSentence += letter;
        updateHistoryUI();
    }
};

if(elements.btnSpace) {
    elements.btnSpace.onclick = () => {
        appState.currentSentence += " ";
        updateHistoryUI();
    };
}
// Evento Pulisci Frase
elements.btnClear.onclick = () => {
    appState.currentSentence = "";
    elements.historyList.innerHTML = "";
    elements.historyList.style.display = 'none';
    elements.historyPlaceholder.style.display = 'flex';
    // Reset timer opzionale
    if (sessionTimer) {
        clearInterval(sessionTimer);
        document.getElementById('session-time').textContent = "0:00";
    }
};