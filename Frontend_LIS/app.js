const BACKEND_URL = 'http://127.0.0.1:5000';
let recognitionActive = false;
let eventSource = null;
let canvasContext = null;
let canvas = null;
let frameCaptureInterval = null;

// Stato applicazione
let appState = {
    isRunning: false,
    isDemo: false,
    signsCount: 0,
    startTime: null,
    timerInterval: null,
    history: [],
    webcamStream: null,
    fps: 0,
    lastFrameTime: 0,
    frameCount: 0 
};

// Elementi DOM
const elements = {
    webcam: document.getElementById('webcam'),
    videoPlaceholder: document.getElementById('video-placeholder'),
    webcamStatus: document.getElementById('webcam-status'),
    btnStart: document.getElementById('btn-start'),
    btnStop: document.getElementById('btn-stop'),
    btnClear: document.getElementById('btn-clear'),
    btnTest: document.getElementById('btn-test'),
    signsCount: document.getElementById('signs-count'),
    sessionTime: document.getElementById('session-time'),
    avgConfidence: document.getElementById('avg-confidence'),
    fpsDisplay: document.getElementById('fps'),
    resultDisplay: document.getElementById('result-display'),
    resultActive: document.getElementById('result-active'),
    resultPlaceholder: document.querySelector('.result-placeholder'),
    detectedLetter: document.getElementById('detected-letter'),
    confidenceFill: document.getElementById('confidence-fill'),
    confidenceValue: document.getElementById('confidence-value'),
    historyList: document.getElementById('history-list'),
    historyPlaceholder: document.getElementById('history-placeholder'),
    outputCanvas: document.getElementById('output-canvas'),
    connectionStatus: document.getElementById('connection-status')
};

// Lettere LIS per demo
// Mappa standard delle connessioni della mano
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
    [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]
];

function drawHandLandmarks(result) {
    const ctx = elements.outputCanvas.getContext('2d');
    ctx.clearRect(0, 0, elements.outputCanvas.width, elements.outputCanvas.height);

    if (!result.has_hand || !result.landmarks) return;

    const landmarks = result.landmarks;

    // 1. Disegna i VETTORI (Linee)
    ctx.strokeStyle = "#10b981"; // Verde
    ctx.lineWidth = 3;
    HAND_CONNECTIONS.forEach(([startIdx, endIdx]) => {
        const start = landmarks[startIdx];
        const end = landmarks[endIdx];
        if (start && end) {
            ctx.beginPath();
            ctx.moveTo(start.x, start.y);
            ctx.lineTo(end.x, end.y);
            ctx.stroke();
        }
    });

    // 2. Disegna i PUNTI (Landmarks)
    landmarks.forEach((point, index) => {
        // Colore originale: bianco con bordo nero, rosso per le punte
        ctx.fillStyle = (index % 4 === 0 && index !== 0) ? "#ff4444" : "#ffffff";
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 1;
        ctx.stroke();
    });

    // 3. Disegna il RETTANGOLO
    if (result.bbox) {
        const [x1, y1, x2, y2] = result.bbox;
        ctx.strokeStyle = "#fbbf24"; // Giallo
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
}

//validare la confidence
function validateConfidence(confidence){
    if(confidence===undefined||confidence===null){
        console.warn('Confidence non fornita, usando default 50%');
        return 50;
    }
    //convertire in numero
    const confNum=parseFLoat(confidence);

    if(isNaN(confNum)){
        console.warn('Confidence non valida:"${confidence}", usando default 50%');
        return 50;
    }
    // Se è in formato 0-1, converti
    if (confNum <= 1.0 && confNum >= 0) {
        return Math.round(confNum * 100);
    }
    
    // Se è già 0-100, limita e arrotonda
    return Math.min(100, Math.max(0, Math.round(confNum)));
}

// Inizializzazione
function init() {

    // Inizializza connection status se esiste
    if (elements.connectionStatus) {
        updateConnectionStatus('Disconnesso', '#ef4444');
    }

    setupEventListeners();
    updateUI();
    console.log('LIS AI Demo inizializzato');
}

// Configura event listeners
function setupEventListeners() {
    elements.btnStart.addEventListener('click', startRecognition);
    elements.btnStop.addEventListener('click', stopRecognition);
    elements.btnClear.addEventListener('click', clearHistory);
    elements.btnTest.addEventListener('click', simulateRecognition);
}

// Avvia riconoscimento
async function startRecognition() {
    if (appState.isRunning) return;
    
    try {
        // Richiedi accesso webcam
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });
        
        elements.webcam.srcObject = stream;
        appState.webcamStream = stream;
        
        elements.webcam.onloadedmetadata = () => {
    // Forza le dimensioni esatte per far combaciare i punti
        elements.outputCanvas.width = 640;
        elements.outputCanvas.height = 480;
        console.log("Canvas dimensionato a 640x480");
        };

        // Crea canvas per catturare frame
        canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 480;
        canvasContext = canvas.getContext('2d');
        
        // Mostra webcam
        elements.webcam.style.display = 'block';
        elements.videoPlaceholder.style.display = 'none';
        
        // Aggiorna stato
        appState.isRunning = true;
        appState.startTime = Date.now();
        appState.signsCount = 0;
        appState.history = [];
        appState.frameCount = 0;
        
        // Avvia timer
        startSessionTimer();
        
        // Avvia FPS counter
        requestAnimationFrame(updateFPS);
        
        // Prova a connettersi al backend
        const backendConnected = await connectToBackend();
        
        if (backendConnected && !appState.isDemo) {
            console.log('Modalità backend attiva');
            // Avvia invio frame al backend ogni 200ms (5 FPS)
            frameCaptureInterval = setInterval(() => {
                captureAndSendFrame();
            }, 200);
        } else {
            console.log('Modalità demo attiva');
            // Per demo, simula riconoscimento ogni 2 secondi
            frameCaptureInterval = setInterval(() => {
                if (Math.random() > 0.7) { // 30% di probabilità
                    simulateRecognition();
                }
            }, 2000);
        }
        
        updateUI();
        console.log('Riconoscimento avviato');
        
    } catch (error) {
        console.error('Errore:', error);
        alert('Impossibile avviare il riconoscimento: ' + error.message);
    }
}

// Cattura e invia un singolo frame
function captureAndSendFrame() {
    if (!appState.isRunning || !canvasContext || appState.isDemo) return;
    
    try {
        // Disegna frame corrente su canvas
        canvasContext.drawImage(elements.webcam, 0, 0, 640, 480);
        
        // Converti in base64 (qualità ridotta per performance)
        const frameData = canvas.toDataURL('image/jpeg', 0.5);
        
        // Invia al backend
        sendFrameToBackend(frameData);
        
    } catch (error) {
        console.error('Errore cattura frame:', error);
    }
}

// Connessione al backend
async function connectToBackend() {
    try {
        console.log('Tentativo di connessione al backend...');
        
        // Test connessione
        const response = await fetch(`${BACKEND_URL}/api/status`);
        const data = await response.json();
        
        if (data.status === 'online') {
            console.log('Backend online, modello dinamico:', data.dynamic_model);
            
            // Aggiorna stato connessione
            updateConnectionStatus('Connesso', '#10b981');
            
            // Connessione SSE per risultati in tempo reale
            eventSource = new EventSource(`${BACKEND_URL}/api/stream`);
            
            eventSource.onopen = () => {
                console.log('Connessione SSE stabilita');
                updateConnectionStatus('Connesso', '#10b981');
            };
            
            eventSource.onmessage = (event) => {
                try {
                    const result = JSON.parse(event.data);
                    
                    // Ignora heartbeat
                    if (result.type === 'heartbeat') {
                        return;
                    }
                    
                    console.log('Ricevuto risultato SSE:', result);
                    handleRecognitionResult({
                        letter: result.letter,
                        confidence: result.confidence ||0, //viene preso dal backend
                        timestamp: new Date().toLocaleTimeString()
                    });
                } catch (e) {
                    console.error('Errore parsing SSE:', e);
                }
            };
            
            eventSource.onerror = (error) => {
                console.error('SSE Error:', error);
                updateConnectionStatus('Errore connessione', '#ef4444');
                // Prova a riconnettersi
                if (eventSource.readyState === EventSource.CLOSED) {
                    console.log('SSE chiuso, tentativo riconnessione...');
                    setTimeout(connectToBackend, 3000);
                }
            };
            
            return true;
        }
    } catch (error) {
        console.error('Errore connessione backend:', error);
        updateConnectionStatus('Disconnesso', '#ef4444');
        
        // Modalità demo fallback
        appState.isDemo = true;
        console.log('Backend non disponibile. Modalità demo attivata.');
        return false;
    }
    return false;
}

// Funzione helper per aggiornare lo stato connessione
function updateConnectionStatus(text, color) {
    if (elements.connectionStatus) {
        const statusText = elements.connectionStatus.querySelector('.status-text');
        const statusDot = elements.connectionStatus.querySelector('.status-dot');
        
        if (statusText) statusText.textContent = text;
        if (statusDot) statusDot.style.backgroundColor = color;
        
        // Aggiorna classi CSS
        elements.connectionStatus.className = 'connection-status';
        if (color === '#10b981') {
            elements.connectionStatus.classList.add('connected');
        } else if (color === '#ef4444') {
            elements.connectionStatus.classList.add('error');
        }
    }
    
    // Aggiorna anche webcamStatus
    if (elements.webcamStatus) {
        if (text === 'Connesso') {
            elements.webcamStatus.textContent = 'Riconoscimento attivo';
            elements.webcamStatus.style.borderColor = '#10b981';
            elements.webcamStatus.style.color = '#10b981';
            elements.webcamStatus.style.background = 'rgba(16, 185, 129, 0.1)';
        } else {
            elements.webcamStatus.textContent = 'In attesa di connessione...';
            elements.webcamStatus.style.borderColor = '#fbbf24';
            elements.webcamStatus.style.color = '#fbbf24';
            elements.webcamStatus.style.background = 'rgba(251, 191, 36, 0.1)';
        }
    }
}

// Cattura e invia frame al backend
function captureAndSendFrames() {
    if (!recognitionActive || !canvasContext) return;
    
    // Disegna frame corrente su canvas
    canvasContext.drawImage(elements.webcam, 0, 0, 640, 480);
    
    // Converti in base64
    const frameData = canvas.toDataURL('image/jpeg', 0.7);
    
    // Invia al backend (ogni 3 frame per ridurre carico)
    if (appState.frameCount % 3 === 0) {
        sendFrameToBackend(frameData);
    }
    
    appState.frameCount++;
    
    // Continua cattura
    if (recognitionActive) {
        setTimeout(captureAndSendFrames, 100); // 10 FPS
    }
}

// Invia frame al backend
async function sendFrameToBackend(frameData) {
    try {
        const response = await fetch(`${BACKEND_URL}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame: frameData })
        });
        
        if (response.ok) {
            const result = await response.json();
            
            // --- AGGIUNTA QUI: Disegna i landmark ---
            drawHandLandmarks(result); 
            // ----------------------------------------

            if (result.letter && !result.error) {
                handleRecognitionResult({
                    letter: result.letter,
                    confidence: result.confidence || 0,
                    timestamp: new Date().toLocaleTimeString()
                });
            }
        }
    } catch (error) {
        console.error("Errore fetch:", error);
    }
}

// Gestisci risultato riconoscimento
function handleRecognitionResult(result) {
    let confidence= result.onfidence|| 0;

    //se confidence è 0-1 converti in percentuale
    if(confidence<=1.0){
    confidence=Math.round(confidence*100);
}
//limita tra 0 e 100
confidence= Math.min(100, Math.max(0, confidence));
    // Aggiorna contatori
    appState.signsCount++;
    
    // Aggiungi alla cronologia
    const recognition = {
        letter: result.letter,
        confidence:confidence, 
        timestamp: new Date().toLocaleTimeString()
    };
    
    appState.history.unshift(recognition);
    if (appState.history.length > 12) {
        appState.history.pop();
    }
    
    // Mostra risultato
    showRecognitionResult(recognition);
    updateUI();
}

// Ferma riconoscimento
function stopRecognition() {
    if (!appState.isRunning) return;
    
    // Ferma cattura frame
    recognitionActive = false;
    
    // Ferma intervallo frame capture
    if (frameCaptureInterval) {
        clearInterval(frameCaptureInterval);
        frameCaptureInterval = null;
    }

    // Chiudi connessione SSE
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    
    // Ferma webcam
    if (appState.webcamStream) {
        appState.webcamStream.getTracks().forEach(track => track.stop());
        appState.webcamStream = null;
    }
    
    // Reset stato
    elements.webcam.style.display = 'none';
    elements.videoPlaceholder.style.display = 'flex';
    
    if (appState.timerInterval) {
        clearInterval(appState.timerInterval);
        appState.timerInterval = null;
    }
    
    appState.isRunning = false;
    elements.connectionStatus.textContent = 'Disconnesso';
    
    updateUI();
    console.log('Riconoscimento fermato');
}

/*
// Simula riconoscimento (demo)
function simulateRecognition() {
    if (!appState.isRunning) {
        alert('Avvia prima il riconoscimento!');
        return;
    }
    
    // Lettera casuale
    const randomLetter = lisLetters[Math.floor(Math.random() * lisLetters.length)];
    const confidence = 70 + Math.random() * 30; // 70-100%
    
    // Aggiorna contatori
    appState.signsCount++;
    
    // Aggiungi alla cronologia
    const recognition = {
        letter: randomLetter,
        confidence: confidence,
        timestamp: new Date().toLocaleTimeString()
    };
    
    appState.history.unshift(recognition);
    if (appState.history.length > 12) {
        appState.history.pop();
    }
    
    // Mostra risultato
    showRecognitionResult(recognition);
    
    // Aggiorna UI
    updateUI();
    
    console.log(`Simulato riconoscimento: ${randomLetter} (${confidence.toFixed(1)}%)`);
}*/

// Mostra risultato del riconoscimento
function showRecognitionResult(result) {
    // Mostra sezione risultati attiva
    elements.resultPlaceholder.style.display = 'none';
    elements.resultActive.style.display = 'block';
    
    // Aggiorna lettera e confidenza
    elements.detectedLetter.textContent = result.letter;
    elements.confidenceValue.textContent = `${result.confidence.toFixed(0)}%`;
    elements.confidenceFill.style.width = `${result.confidence}%`;
    
    // Animazione
    elements.detectedLetter.classList.add('slide-in');
    setTimeout(() => {
        elements.detectedLetter.classList.remove('slide-in');
    }, 500);
    
    // Aggiorna cronologia
    updateHistory();
}

// Aggiorna cronologia UI
function updateHistory() {
    if (appState.history.length === 0) {
        elements.historyPlaceholder.style.display = 'flex';
        elements.historyList.style.display = 'none';
        return;
    }
    
    elements.historyPlaceholder.style.display = 'none';
    elements.historyList.style.display = 'flex';
    
    // Crea elementi cronologia
    elements.historyList.innerHTML = '';
    appState.history.forEach((item, index) => {
        const historyItem = document.createElement('div');
        historyItem.className = `history-item ${index === 0 ? 'latest' : ''}`;
        historyItem.textContent = item.letter;
        historyItem.title = `${item.letter} - ${item.confidence.toFixed(0)}% (${item.timestamp})`;
        elements.historyList.appendChild(historyItem);
    });
}

// Pulisci cronologia
function clearHistory() {
    appState.history = [];
    appState.signsCount = 0;
    
    // Ripristina placeholder
    elements.resultPlaceholder.style.display = 'flex';
    elements.resultActive.style.display = 'none';
    elements.historyPlaceholder.style.display = 'flex';
    elements.historyList.style.display = 'none';
    
    updateUI();
    console.log('Cronologia pulita');
}

// Avvia timer sessione
function startSessionTimer() {
    appState.timerInterval = setInterval(() => {
        if (appState.startTime) {
            const elapsed = Date.now() - appState.startTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            elements.sessionTime.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
    }, 1000);
}

// Calcola FPS
function updateFPS(timestamp) {
    if (appState.lastFrameTime) {
        const delta = timestamp - appState.lastFrameTime;
        appState.fps = Math.round(1000 / delta);
        elements.fpsDisplay.textContent = appState.fps;
    }
    appState.lastFrameTime = timestamp;
    
    if (appState.isRunning) {
        requestAnimationFrame(updateFPS);
    }
}

// Simula connessione backend
function simulateBackendConnection() {
    elements.connectionStatus.textContent = 'Connesso';
    elements.connectionStatus.style.color = '#10b981';
    elements.webcamStatus.textContent = 'Riconoscimento attivo';
    elements.webcamStatus.style.borderColor = '#10b981';
    elements.webcamStatus.style.color = '#10b981';
    elements.webcamStatus.style.background = 'rgba(16, 185, 129, 0.1)';
}

// Aggiorna UI in base allo stato
function updateUI() {
    // Contatori
    elements.signsCount.textContent = appState.signsCount;
    
    // Confidenza media
    if (appState.history.length > 0) {
        const avgConf = appState.history.reduce((sum, item) => sum + item.confidence, 0) / appState.history.length;
        elements.avgConfidence.textContent = `${avgConf.toFixed(0)}%`;
    } else {
        elements.avgConfidence.textContent = '0%';
    }
    
    // Stato bottoni
    elements.btnStart.disabled = appState.isRunning;
    elements.btnStop.disabled = !appState.isRunning;
    elements.btnTest.disabled = !appState.isRunning;
    
    // Stato connessione
    if (!appState.isRunning) {
        elements.connectionStatus.textContent = 'Disconnesso';
        elements.connectionStatus.style.color = '#ef4444';
        elements.webcamStatus.textContent = 'In attesa di avvio...';
        elements.webcamStatus.style.borderColor = '#fbbf24';
        elements.webcamStatus.style.color = '#fbbf24';
        elements.webcamStatus.style.background = 'rgba(251, 191, 36, 0.1)';
    }
}

// Inizializza app quando il DOM è pronto
document.addEventListener('DOMContentLoaded', init);

// Tasti rapidi
document.addEventListener('keydown', (e) => {
    if (e.key === ' ' || e.key === 'Spacebar') { // Spazio
        if (appState.isRunning) {
            simulateRecognition();
        }
        e.preventDefault();
    } else if (e.key === 'Escape') { // ESC
        stopRecognition();
    } else if (e.key === 't' || e.key === 'T') { // T
        if (appState.isRunning) {
            simulateRecognition();
        }
    }
});