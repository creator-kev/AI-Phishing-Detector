// Popup script for AI Phishing Detector Extension

const API_URL = 'http://127.0.0.1:5000/api/predict';

// DOM Elements
const statusIcon = document.getElementById('statusIcon');
const statusTitle = document.getElementById('statusTitle');
const statusDescription = document.getElementById('statusDescription');
const currentUrl = document.getElementById('currentUrl');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceValue = document.getElementById('confidenceValue');
const progressFill = document.getElementById('progressFill');
const recheckBtn = document.getElementById('recheckBtn');
const detailsBtn = document.getElementById('detailsBtn');
const statusCard = document.getElementById('statusCard');
const checkedCount = document.getElementById('checkedCount');
const blockedCount = document.getElementById('blockedCount');

// Toggle buttons
const notificationToggle = document.getElementById('notificationToggle');
const autoCheckToggle = document.getElementById('autoCheckToggle');
const warningsToggle = document.getElementById('warningsToggle');

let currentTabUrl = '';
let lastResult = null;

// Initialize popup
async function initialize() {
    // Load settings
    loadSettings();
    
    // Load stats
    loadStats();
    
    // Get current tab URL
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tabs[0]) {
        currentTabUrl = tabs[0].url;
        currentUrl.textContent = currentTabUrl;
        
        // Check if auto-check is enabled
        const settings = await chrome.storage.sync.get(['autoCheck']);
        if (settings.autoCheck !== false) {
            checkURL(currentTabUrl);
        }
    }
}

// Check URL for phishing
async function checkURL(url) {
    try {
        // Update UI to checking state
        updateUI('checking');
        
        // Call API
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: url })
        });
        
        if (!response.ok) {
            throw new Error('API request failed');
        }
        
        const result = await response.json();
        lastResult = result;
        
        // Update stats
        incrementStat('checked');
        if (result.is_phishing) {
            incrementStat('blocked');
        }
        
        // Update UI with result
        updateUI(result.is_phishing ? 'danger' : 'safe', result);
        
        // Send notification if enabled
        const settings = await chrome.storage.sync.get(['notifications']);
        if (settings.notifications !== false && result.is_phishing) {
            chrome.notifications.create({
                type: 'basic',
                iconUrl: 'icons/icon48.png',
                title: '⚠️ Phishing Detected!',
                message: `This website might be trying to steal your information.\n\nConfidence: ${(result.confidence).toFixed(1)}%`,
                priority: 2
            });
        }
        
    } catch (error) {
        console.error('Error checking URL:', error);
        updateUI('error');
    }
}

// Update UI based on status
function updateUI(status, data = null) {
    switch (status) {
        case 'checking':
            statusIcon.textContent = '🔍';
            statusTitle.textContent = 'Analyzing URL...';
            statusDescription.textContent = 'Please wait';
            confidenceBar.style.display = 'none';
            statusCard.style.background = 'white';
            break;
            
        case 'safe':
            statusIcon.textContent = '✅';
            statusTitle.textContent = 'Safe Website';
            statusTitle.style.color = '#28a745';
            statusDescription.textContent = 'No threats detected';
            confidenceBar.style.display = 'block';
            confidenceValue.textContent = `${(data.confidence).toFixed(1)}%`;
            progressFill.style.width = `${100 - data.confidence}%`;
            progressFill.className = 'progress-fill progress-safe';
            statusCard.style.background = 'linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%)';
            break;
            
        case 'danger':
            statusIcon.textContent = '⚠️';
            statusTitle.textContent = 'Phishing Detected!';
            statusTitle.style.color = '#dc3545';
            statusDescription.textContent = 'Do not enter personal information';
            confidenceBar.style.display = 'block';
            confidenceValue.textContent = `${(data.confidence).toFixed(1)}%`;
            progressFill.style.width = `${data.confidence}%`;
            progressFill.className = 'progress-fill progress-danger';
            statusCard.style.background = 'linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%)';
            break;
            
        case 'error':
            statusIcon.textContent = '❌';
            statusTitle.textContent = 'Check Failed';
            statusTitle.style.color = '#ffc107';
            statusDescription.textContent = 'Could not connect to API';
            confidenceBar.style.display = 'none';
            statusCard.style.background = 'white';
            break;
    }
}

// Load settings from storage
async function loadSettings() {
    const settings = await chrome.storage.sync.get(['notifications', 'autoCheck', 'warnings']);
    
    notificationToggle.classList.toggle('active', settings.notifications !== false);
    autoCheckToggle.classList.toggle('active', settings.autoCheck !== false);
    warningsToggle.classList.toggle('active', settings.warnings !== false);
}

// Load stats from storage
async function loadStats() {
    const stats = await chrome.storage.local.get(['checked', 'blocked']);
    checkedCount.textContent = stats.checked || 0;
    blockedCount.textContent = stats.blocked || 0;
}

// Increment stat counter
async function incrementStat(type) {
    const stats = await chrome.storage.local.get([type]);
    const newValue = (stats[type] || 0) + 1;
    await chrome.storage.local.set({ [type]: newValue });
    
    if (type === 'checked') {
        checkedCount.textContent = newValue;
    } else if (type === 'blocked') {
        blockedCount.textContent = newValue;
    }
}

// Event Listeners
recheckBtn.addEventListener('click', () => {
    if (currentTabUrl) {
        checkURL(currentTabUrl);
    }
});

detailsBtn.addEventListener('click', () => {
    if (lastResult) {
        alert(`Detailed Analysis:\n\nVerdict: ${lastResult.verdict}\nProbability: ${lastResult.phishing_probability}\nConfidence: ${lastResult.confidence}%\n\nModel: ${lastResult.model}\nFeatures: ${lastResult.features}`);
    }
});

document.getElementById('openDashboard').addEventListener('click', (e) => {
    e.preventDefault();
    chrome.tabs.create({ url: 'http://127.0.0.1:5000' });
});

document.getElementById('reportIssue').addEventListener('click', (e) => {
    e.preventDefault();
    alert('Report functionality coming soon!');
});

// Toggle settings
notificationToggle.addEventListener('click', async () => {
    notificationToggle.classList.toggle('active');
    const isActive = notificationToggle.classList.contains('active');
    await chrome.storage.sync.set({ notifications: isActive });
});

autoCheckToggle.addEventListener('click', async () => {
    autoCheckToggle.classList.toggle('active');
    const isActive = autoCheckToggle.classList.contains('active');
    await chrome.storage.sync.set({ autoCheck: isActive });
});

warningsToggle.addEventListener('click', async () => {
    warningsToggle.classList.toggle('active');
    const isActive = warningsToggle.classList.contains('active');
    await chrome.storage.sync.set({ warnings: isActive });
});

// Initialize on load
initialize();
