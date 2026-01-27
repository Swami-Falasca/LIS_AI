// Stato applicazione
let appState = {
    isRunning: false,
    isDemo: true,
    signsCount: 0,
    startTime: null,
    timerInterval: null,
    history: [],
    webcamStream: null,
    fps: 0,
    lastFrameTime: 0
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
    connectionStatus: document.getElementById('connection-status')
};

// Lettere LIS per demo
const lisLetters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

// Inizializzazione
function init() {
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

// Avvia riconoscimento (demo)
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
        elements.webcam.style.display = 'block';
        elements.videoPlaceholder.style.display = 'none';
        
        // Aggiorna stato
        appState.isRunning = true;
        appState.startTime = Date.now();
        appState.signsCount = 0;
        appState.history = [];
        
        // Avvia timer
        startSessionTimer();
        
        // Avvia FPS counter
        requestAnimationFrame(updateFPS);
        
        // Simula riconnessione backend
        simulateBackendConnection();
        
        updateUI();
        console.log('Riconoscimento avviato');
        
    } catch (error) {
        console.error('Errore accesso webcam:', error);
        alert('Impossibile accedere alla webcam. Controlla i permessi.');
    }
}

// Ferma riconoscimento
function stopRecognition() {
    if (!appState.isRunning) return;
    
    // Ferma webcam
    if (appState.webcamStream) {
        appState.webcamStream.getTracks().forEach(track => track.stop());
        appState.webcamStream = null;
    }
    
    elements.webcam.style.display = 'none';
    elements.videoPlaceholder.style.display = 'flex';
    
    // Ferma timer
    if (appState.timerInterval) {
        clearInterval(appState.timerInterval);
        appState.timerInterval = null;
    }
    
    // Aggiorna stato
    appState.isRunning = false;
    appState.connectionStatus = 'Disconnesso';
    
    updateUI();
    console.log('Riconoscimento fermato');
}

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
}

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
