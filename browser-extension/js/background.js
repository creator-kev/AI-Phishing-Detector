// Background service worker for AI Phishing Detector

const API_URL = 'http://127.0.0.1:5000/api/predict';
const cache = new Map();
const CACHE_DURATION = 1000 * 60 * 30; // 30 minutes

// Listen for tab updates
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete' && tab.url) {
        // Check if auto-check is enabled
        const settings = await chrome.storage.sync.get(['autoCheck', 'warnings']);
        
        if (settings.autoCheck !== false) {
            checkURLInBackground(tab.url, tabId, settings.warnings !== false);
        }
    }
});

// Check URL in background
async function checkURLInBackground(url, tabId, showWarnings) {
    // Skip non-http(s) URLs
    if (!url.startsWith('http')) {
        return;
    }
    
    // Check cache first
    const cached = cache.get(url);
    if (cached && (Date.now() - cached.timestamp < CACHE_DURATION)) {
        if (cached.isPhishing && showWarnings) {
            injectWarning(tabId, cached.data);
        }
        return;
    }
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: url })
        });
        
        if (!response.ok) {
            return;
        }
        
        const result = await response.json();
        
        // Cache result
        cache.set(url, {
            data: result,
            isPhishing: result.is_phishing,
            timestamp: Date.now()
        });
        
        // Show warning if phishing detected
        if (result.is_phishing && showWarnings) {
            injectWarning(tabId, result);
            
            // Update badge
            chrome.action.setBadgeText({ text: '⚠️', tabId: tabId });
            chrome.action.setBadgeBackgroundColor({ color: '#dc3545', tabId: tabId });
        } else {
            chrome.action.setBadgeText({ text: '', tabId: tabId });
        }
        
    } catch (error) {
        console.error('Background check failed:', error);
    }
}

// Inject warning into page
function injectWarning(tabId, data) {
    chrome.scripting.executeScript({
        target: { tabId: tabId },
        func: showWarningBanner,
        args: [data]
    });
}

// Function to show warning banner (injected into page)
function showWarningBanner(data) {
    // Remove existing warning if any
    const existing = document.getElementById('ai-phishing-warning');
    if (existing) {
        existing.remove();
    }
    
    // Create warning banner
    const warning = document.createElement('div');
    warning.id = 'ai-phishing-warning';
    warning.innerHTML = `
        <div style="position: fixed; top: 0; left: 0; right: 0; background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); color: white; padding: 20px; z-index: 999999; box-shadow: 0 5px 20px rgba(0,0,0,0.3); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            <div style="max-width: 1200px; margin: 0 auto; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 15px;">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <span style="font-size: 32px;">⚠️</span>
                    <div>
                        <h3 style="margin: 0; font-size: 18px; font-weight: bold;">Phishing Warning</h3>
                        <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;">This website may be trying to steal your personal information. Confidence: ${data.confidence.toFixed(1)}%</p>
                    </div>
                </div>
                <div style="display: flex; gap: 10px;">
                    <button onclick="this.parentElement.parentElement.parentElement.remove()" style="background: white; color: #dc3545; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 14px;">Dismiss</button>
                    <button onclick="window.history.back()" style="background: rgba(255,255,255,0.2); color: white; border: 2px solid white; padding: 10px 20px; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 14px;">← Go Back</button>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertBefore(warning, document.body.firstChild);
}

// Clean old cache entries periodically
setInterval(() => {
    const now = Date.now();
    for (const [url, data] of cache.entries()) {
        if (now - data.timestamp > CACHE_DURATION) {
            cache.delete(url);
        }
    }
}, 1000 * 60 * 10); // Clean every 10 minutes
