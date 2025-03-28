// --- Configuration ---
// Should match the port in config.py (UI_WEBSOCKET_PORT)
const WEBSOCKET_PORT = 8766; // Default, adjust if config changes
const WS_URL = `ws://localhost:${WEBSOCKET_PORT}`;
const RECONNECT_TIMEOUT = 5000; // 5 seconds

// --- DOM Elements ---
const mainCircle = document.getElementById('main-circle');
const micIcon = document.getElementById('mic-icon');
const oscillatingCircles = document.getElementById('oscillating-circles');
const transcriptionArea = document.getElementById('transcription-area');
const userTextElement = document.getElementById('user-text');
const aiTextElement = document.getElementById('ai-text');
const toggleTextBtn = document.getElementById('toggle-text-btn');
const themeButtons = document.querySelectorAll('.theme-btn');
const bodyElement = document.body;

// --- State ---
let websocket = null;
let isAiSpeaking = false;
let isUserSpeaking = false;
let isTextVisible = false; // Text area starts hidden (matching CSS)
let reconnectTimer = null; // To store the reconnect timeout ID

// --- WebSocket Logic ---

function connectWebSocket() {
    // Clear any existing reconnect timer
    if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
    }

    console.log(`Attempting to connect to WebSocket at ${WS_URL}...`);
    websocket = new WebSocket(WS_URL);

    websocket.onopen = () => {
        console.log('WebSocket connection established.');
        // Optional: Send a message to confirm connection if needed
        // websocket.send(JSON.stringify({ type: "ui_connected", client: "web_ui" }));
    };

    websocket.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            // console.log('Message from server:', message); // Debugging
            handleWebSocketMessage(message);
        } catch (error) {
            console.error('Failed to parse WebSocket message:', event.data, error);
        }
    };

    websocket.onerror = (error) => {
        // This event doesn't provide detailed error info, just that an error occurred.
        console.error('WebSocket error occurred. Connection will likely close.');
        // The onclose event will handle cleanup and reconnection attempts.
    };

    websocket.onclose = (event) => {
        console.warn(`WebSocket closed. Code: ${event.code}, Reason: ${event.reason || 'No reason given'}.`);
        websocket = null; // Clear the instance

        // Reset speaking states on disconnect for clarity
        setAiSpeaking(false);
        setUserSpeaking(false);

        // Attempt to reconnect only if not explicitly closed cleanly (e.g., code 1000)
        // Codes like 1006 (Abnormal Closure) usually indicate connection issues.
        if (event.code !== 1000) {
            console.log(`Attempting reconnect in ${RECONNECT_TIMEOUT / 1000}s...`);
            if (reconnectTimer) clearTimeout(reconnectTimer); // Clear existing timer just in case
            reconnectTimer = setTimeout(connectWebSocket, RECONNECT_TIMEOUT);
        } else {
            console.log("WebSocket closed cleanly.");
        }
    };
}

function handleWebSocketMessage(message) {
    if (!message || typeof message.type === 'undefined') {
        console.warn('Received invalid message format:', message);
        return;
    }

    switch (message.type) {
        case 'state_update':
            if (typeof message.state !== 'string') {
                console.warn('Invalid state_update message:', message);
                break;
            }
            handleStateUpdate(message.state);
            break;
        case 'text_update':
            if (typeof message.role !== 'string' || typeof message.text !== 'string') {
                console.warn('Invalid text_update message:', message);
                break;
            }
            handleTextUpdate(message.role, message.text);
            break;
        // Add cases for other message types if needed (e.g., 'clear_text', 'service_status')
        default:
            console.warn('Received unknown message type:', message.type);
    }
}

function handleStateUpdate(state) {
    switch (state) {
        case 'ai_speaking_start':
            setAiSpeaking(true);
            // Ensure user speaking indicator is off if AI starts
            setUserSpeaking(false);
            break;
        case 'ai_speaking_stop':
            setAiSpeaking(false);
            break;
        case 'user_speaking_start':
            setUserSpeaking(true);
            // Ensure AI speaking indicator is off if user starts
            setAiSpeaking(false);
            break;
        case 'user_speaking_stop':
            setUserSpeaking(false);
            break;
        default:
            console.warn('Unknown state received:', state);
    }
}

function handleTextUpdate(role, text) {
    // Ensure text area is visible when text updates arrive
    if (!isTextVisible) {
        toggleTextVisibility(); // Show the text area
    }

    if (role === 'user') {
        userTextElement.textContent = text;
        // Optionally clear AI text when user speaks/types
        // aiTextElement.textContent = '';
    } else if (role === 'ai') {
        aiTextElement.textContent = text;
        // Optionally clear user text when AI responds
        // userTextElement.textContent = '';
    } else {
        console.warn('Unknown role in text_update:', role);
    }

    // Basic scroll into view (optional, could be improved)
    // transcriptionArea.scrollTop = transcriptionArea.scrollHeight;
    if (role === 'ai' && aiTextElement.scrollIntoView) {
        aiTextElement.scrollIntoView({ behavior: 'smooth', block: 'end' });
    } else if (role === 'user' && userTextElement.scrollIntoView) {
        userTextElement.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
}

// --- UI Update Functions ---

function setAiSpeaking(speaking) {
    if (isAiSpeaking === speaking) return; // Avoid redundant updates
    isAiSpeaking = speaking;

    if (speaking) {
        console.log("UI Update: AI Speaking START");
        mainCircle.classList.add('pulsing');
    } else {
        console.log("UI Update: AI Speaking STOP");
        mainCircle.classList.remove('pulsing');
    }
}

function setUserSpeaking(speaking) {
    if (isUserSpeaking === speaking) return; // Avoid redundant updates
    isUserSpeaking = speaking;

    if (speaking) {
        console.log("UI Update: User Speaking START");
        micIcon.classList.remove('active');
        oscillatingCircles.classList.add('active');
        // TODO: Implement volume sensitivity visualization here if desired
    } else {
        console.log("UI Update: User Speaking STOP");
        oscillatingCircles.classList.remove('active');
        micIcon.classList.add('active');
    }
}

function toggleTextVisibility() {
    isTextVisible = !isTextVisible;
    if (isTextVisible) {
        transcriptionArea.classList.remove('hidden');
        toggleTextBtn.textContent = '📝'; // Indicate text is visible (e.g., notepad with text)
        toggleTextBtn.title = "Hide Transcription";
        console.log("UI Action: Show Transcription");
    } else {
        transcriptionArea.classList.add('hidden');
        toggleTextBtn.textContent = '📄'; // Indicate text is hidden (e.g., blank page)
        toggleTextBtn.title = "Show Transcription";
        console.log("UI Action: Hide Transcription");
    }
}

function applyTheme(themeClass) {
    // Remove any existing theme classes from body (safer than setting className = '')
    bodyElement.classList.remove(
        'theme-dark',
        'theme-light',
        'theme-blue',
        'theme-purple',
        'theme-red'
        // Add any other theme classes here if created
    );
    // Add the new theme class
    if (themeClass) {
        bodyElement.classList.add(themeClass);
        console.log("Applied theme:", themeClass);
        // Save theme choice to localStorage for persistence
        try {
            localStorage.setItem('cypher_theme', themeClass);
        } catch (e) {
            console.warn("Could not save theme to localStorage:", e);
        }
    }
}

// --- Event Listeners ---

// Toggle transcription visibility
if (toggleTextBtn) {
    toggleTextBtn.addEventListener('click', toggleTextVisibility);
} else {
    console.error("Could not find #toggle-text-btn element.");
}

// Apply theme on button click
themeButtons.forEach(button => {
    button.addEventListener('click', () => {
        const theme = button.getAttribute('data-theme');
        if (theme) {
            applyTheme(theme);
        } else {
            console.warn("Theme button clicked without data-theme attribute:", button);
        }
    });
});

// --- Initialization ---

document.addEventListener('DOMContentLoaded', () => {
    console.log('UI Initialized. DOM fully loaded.');

    // 1. Load saved theme or apply default
    let initialTheme = 'theme-dark'; // Default theme
    try {
        const savedTheme = localStorage.getItem('cypher_theme');
        if (savedTheme && bodyElement.classList.contains(savedTheme)) { // Check if valid theme
            initialTheme = savedTheme;
            console.log("Loaded saved theme:", initialTheme);
        } else if (savedTheme) {
             console.warn(`Saved theme "${savedTheme}" not recognized, using default.`);
             localStorage.removeItem('cypher_theme'); // Remove invalid theme
        }
    } catch (e) {
        console.warn("Could not load theme from localStorage:", e);
    }
    applyTheme(initialTheme); // Apply the determined theme


    // 2. Set initial visual state based on CSS defaults
    // Ensure JS state matches initial CSS state (text hidden, mic active)
    isTextVisible = !transcriptionArea.classList.contains('hidden'); // Sync state
    if (isTextVisible) { // Update button if text starts visible (unlikely based on HTML)
         toggleTextBtn.textContent = '📝';
         toggleTextBtn.title = "Hide Transcription";
    } else {
         toggleTextBtn.textContent = '📄';
         toggleTextBtn.title = "Show Transcription";
    }

    isAiSpeaking = mainCircle.classList.contains('pulsing');
    isUserSpeaking = oscillatingCircles.classList.contains('active');
    if (!isUserSpeaking) {
        micIcon.classList.add('active'); // Ensure mic is active if user isn't speaking
    }

    console.log(`Initial UI State: Text Visible=${isTextVisible}, AI Speaking=${isAiSpeaking}, User Speaking=${isUserSpeaking}`);

    // 3. Start WebSocket connection
    connectWebSocket();
});