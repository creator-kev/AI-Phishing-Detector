# 🛡️ AI Phishing Detector - API Documentation

## Base URL
```
http://127.0.0.1:5000
```

## Authentication

### API Key Authentication
Add to headers:
```
X-API-Key: your-api-key-here
```

### Generate API Key
```bash
POST /api/key/generate
Content-Type: application/json

{
  "user_id": "your_user_id"
}
```

**Response:**
```json
{
  "api_key": "pk_abc123...",
  "user_id": "your_user_id",
  "created_at": "2024-01-01T00:00:00",
  "note": "Save this key securely"
}
```

---

## Endpoints

### 1. Single URL Prediction

**Endpoint:** `POST /api/predict`

**Rate Limit:** 50 requests/hour

**Headers:**
```
Content-Type: application/json
X-API-Key: your-api-key (if auth enabled)
```

**Request:**
```json
{
  "text": "http://suspicious-site.tk/login"
}
```

**Response:**
```json
{
  "verdict": "PHISHING",
  "phishing_probability": 0.9234,
  "is_phishing": true,
  "is_legitimate": false,
  "confidence": 92.34,
  "threshold": 0.5,
  "timestamp": "2024-01-01T12:00:00",
  "url_analyzed": "http://suspicious-site.tk/login",
  "model": "RandomForestClassifier",
  "features": "37 features",
  "processing_time_ms": 45.23
}
```

**cURL Example:**
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"text":"http://example.com"}'
```

**Python Example:**
```python
import requests

url = "http://127.0.0.1:5000/api/predict"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your-key"
}
data = {"text": "http://suspicious-site.com"}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

---

### 2. Batch URL Prediction

**Endpoint:** `POST /api/batch`

**Rate Limit:** 10 requests/hour

**Request:**
```json
{
  "urls": [
    "https://google.com",
    "http://phishing-site.tk",
    "https://github.com"
  ]
}
```

**Response:**
```json
{
  "total": 3,
  "results": [
    {
      "verdict": "LEGITIMATE",
      "phishing_probability": 0.0234,
      "confidence": 2.34,
      ...
    },
    {
      "verdict": "PHISHING",
      "phishing_probability": 0.9567,
      "confidence": 95.67,
      ...
    },
    ...
  ],
  "timestamp": "2024-01-01T12:00:00"
}
```

**Limits:**
- Maximum 100 URLs per request
- Each URL must be valid HTTP/HTTPS

---

### 3. Health Check

**Endpoint:** `GET /health`

**Rate Limit:** None

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "features": 37,
  "timestamp": "2024-01-01T12:00:00",
  "version": "2.0-enhanced"
}
```

---

### 4. API Statistics

**Endpoint:** `GET /api/stats`

**Response:**
```json
{
  "features": 37,
  "model": "RandomForestClassifier",
  "authentication": "enabled",
  "rate_limiting": "enabled",
  "caching": "enabled",
  "version": "2.0"
}
```

---

### 5. Monitoring Dashboard

**Endpoint:** `GET /dashboard`

Web-based monitoring dashboard with:
- Real-time request statistics
- Phishing detection metrics
- Response time graphs
- Activity logs

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/api/predict` | 50 requests/hour |
| `/api/batch` | 10 requests/hour |
| `/api/key/generate` | 5 requests/hour |
| General | 100 requests/hour |

**Rate Limit Headers:**
```
X-RateLimit-Limit: 50
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640000000
```

When exceeded:
```json
{
  "error": "Rate limit exceeded",
  "message": "Too many requests. Try again later."
}
```

---

## Caching

Responses are cached for **30 minutes** based on URL.

**Cache Headers:**
```
X-Cache: HIT  (or MISS)
```

Benefits:
- Faster response times for repeated URLs
- Reduced server load
- Lower API call costs

---

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Missing API key |
| 403 | Forbidden - Invalid API key |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |

**Error Response Format:**
```json
{
  "error": "Error type",
  "message": "Detailed error message"
}
```

---

## Best Practices

### 1. Use Caching
Repeated URLs are cached. Store results client-side when possible.

### 2. Batch Requests
Use `/api/batch` for multiple URLs instead of multiple `/api/predict` calls.

### 3. Handle Rate Limits
Implement exponential backoff when receiving 429 responses.

### 4. Validate URLs
Pre-validate URLs client-side before sending to API.

### 5. Secure API Keys
- Never commit API keys to version control
- Use environment variables
- Rotate keys regularly

---

## SDK Examples

### JavaScript/Node.js
```javascript
const axios = require('axios');

async function checkURL(url) {
  try {
    const response = await axios.post('http://127.0.0.1:5000/api/predict', {
      text: url
    }, {
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': process.env.API_KEY
      }
    });
    
    return response.data;
  } catch (error) {
    if (error.response?.status === 429) {
      console.log('Rate limit exceeded. Waiting...');
      await new Promise(resolve => setTimeout(resolve, 60000));
      return checkURL(url);
    }
    throw error;
  }
}
```

### Python
```python
import requests
from time import sleep

class PhishingDetector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://127.0.0.1:5000"
    
    def check_url(self, url):
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }
        
        response = requests.post(
            f"{self.base_url}/api/predict",
            json={"text": url},
            headers=headers
        )
        
        if response.status_code == 429:
            print("Rate limit exceeded. Waiting...")
            sleep(60)
            return self.check_url(url)
        
        response.raise_for_status()
        return response.json()

# Usage
detector = PhishingDetector("your-api-key")
result = detector.check_url("http://example.com")
print(f"Verdict: {result['verdict']}")
```

---

## Interactive Documentation

Access Swagger UI for interactive API testing:
```
http://127.0.0.1:5000/apidocs
```

Features:
- Try API endpoints directly in browser
- View request/response schemas
- Generate code samples
- Test authentication

---

## Support

- 📧 Email: support@example.com
- 📖 Full Docs: http://127.0.0.1:5000/apidocs
- 🐛 Issues: GitHub Issues
- 💬 Discord: [Join Server](#)

---

## Changelog

### v2.0 (Enhanced)
- ✅ Added rate limiting
- ✅ Added API key authentication
- ✅ Added response caching
- ✅ Added comprehensive logging
- ✅ Added Swagger documentation
- ✅ Added monitoring dashboard
- ✅ Improved error handling

### v1.0 (Initial)
- ✅ Basic ML prediction
- ✅ Single URL endpoint
- ✅ Batch processing
