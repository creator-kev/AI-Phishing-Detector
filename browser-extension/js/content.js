// Content script for AI Phishing Detector
// Runs on all pages to provide real-time protection

console.log('🛡️ AI Phishing Detector: Active');

// Monitor for suspicious form submissions
document.addEventListener('submit', async (e) => {
    const form = e.target;
    
    // Check if form has password fields
    const hasPassword = form.querySelector('input[type="password"]');
    const hasEmail = form.querySelector('input[type="email"], input[name*="email"]');
    
    if (hasPassword || hasEmail) {
        // Get current URL
        const currentUrl = window.location.href;
        
        // Check with extension
        chrome.runtime.sendMessage({
            action: 'checkBeforeSubmit',
            url: currentUrl
        }, (response) => {
            if (response && response.isPhishing) {
                e.preventDefault();
                if (confirm('⚠️ WARNING: This site may be a phishing attempt!\n\nAre you sure you want to submit your credentials?')) {
                    form.submit();
                }
            }
        });
    }
});
