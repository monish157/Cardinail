#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>

// Access Point credentials
const char* ap_ssid = "NailScanner";
const char* ap_password = "12345678";

// Create web server on port 80
WebServer server(80);

// LED pin (built-in LED)
const int ledPin = 2;

// Variables to store latest prediction data
String currentPrediction = "Waiting for connection...";
float currentConfidence = 0.0;
String statusMessage = "ESP32 Ready";
unsigned long lastUpdate = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  // Initialize LED
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW);
  
  // Configure Access Point
  Serial.println("Setting up Access Point...");
  WiFi.softAP(ap_ssid, ap_password);
  
  IPAddress IP = WiFi.softAPIP();
  Serial.print("Access Point IP address: ");
  Serial.println(IP);
  Serial.println("WiFi Hotspot Details:");
  Serial.println("SSID: " + String(ap_ssid));
  Serial.println("Password: " + String(ap_password));
  Serial.println("Connect your laptop and mobile to this network");
  
  // Setup mDNS
  if (MDNS.begin("nailscanner")) {
    Serial.println("mDNS responder started");
    Serial.println("Access via: http://nailscanner.local");
  }
  
  // Turn on LED when ready
  digitalWrite(ledPin, HIGH);
  
  // Setup web server routes
  server.on("/", handleRoot);
  server.on("/mobile", handleMobile);
  server.on("/api/status", handleAPIStatus);
  server.on("/api/update", HTTP_POST, handleUpdatePrediction);
  server.on("/video_feed", handleVideoProxy);
  server.onNotFound(handleNotFound);
  
  // Start server
  server.begin();
  Serial.println("ESP32 web server started");
  Serial.println("Laptop: Open browser and go to http://192.168.4.1");
  Serial.println("Mobile: Open browser and go to http://192.168.4.1/mobile");
  
  // LED indication pattern
  blinkLED(3, 200); // 3 quick blinks to indicate ready
}

void loop() {
  // Handle web server requests
  server.handleClient();
  
  // LED heartbeat
  static unsigned long lastBlink = 0;
  if (millis() - lastBlink > 2000) {
    digitalWrite(ledPin, !digitalRead(ledPin));
    lastBlink = millis();
  }
  
  delay(10);
}

void handleRoot() {
  String html = generateLaptopHTML();
  server.send(200, "text/html", html);
}

void handleMobile() {
  String html = generateMobileHTML();
  server.send(200, "text/html", html);
}

void handleAPIStatus() {
  String json = "{";
  json += "\"prediction\":\"" + currentPrediction + "\",";
  json += "\"confidence\":" + String(currentConfidence) + ",";
  json += "\"status\":\"" + statusMessage + "\",";
  json += "\"timestamp\":" + String(millis()) + ",";
  json += "\"uptime\":" + String(millis()/1000);
  json += "}";
  
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", json);
}

void handleUpdatePrediction() {
  if (server.hasArg("plain")) {
    String body = server.arg("plain");
    
    // Simple JSON parsing (basic implementation)
    if (body.indexOf("prediction") > -1) {
      int start = body.indexOf("prediction\":\"") + 13;
      int end = body.indexOf("\"", start);
      if (start > 12 && end > start) {
        currentPrediction = body.substring(start, end);
      }
    }
    
    if (body.indexOf("confidence") > -1) {
      int start = body.indexOf("confidence\":") + 12;
      int end = body.indexOf(",", start);
      if (end == -1) end = body.indexOf("}", start);
      if (start > 11 && end > start) {
        currentConfidence = body.substring(start, end).toFloat();
      }
    }
    
    lastUpdate = millis();
    statusMessage = "Updated from laptop";
    
    Serial.println("Received update: " + currentPrediction + " (" + String(currentConfidence*100) + "%)");
    
    server.send(200, "text/plain", "OK");
  } else {
    server.send(400, "text/plain", "Bad Request");
  }
}

void handleVideoProxy() {
  // This will redirect to laptop's video feed if available
  server.sendHeader("Location", "http://192.168.4.2:5000/video_feed");
  server.send(302, "text/plain", "Redirecting to video feed");
}

void handleNotFound() {
  String message = "File Not Found\n\n";
  message += "URI: " + server.uri() + "\n";
  message += "Method: " + (server.method() == HTTP_GET ? "GET" : "POST") + "\n";
  message += "Arguments: " + server.args() + "\n";
  server.send(404, "text/plain", message);
}

void blinkLED(int times, int delayMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(ledPin, HIGH);
    delay(delayMs);
    digitalWrite(ledPin, LOW);
    delay(delayMs);
  }
}

String generateLaptopHTML() {
  return R"(
<!DOCTYPE html>
<html>
<head>
    <title>NailScanner - Laptop Interface</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .video-section {
            text-align: center;
            margin: 20px 0;
        }
        .status-panel {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .btn {
            background: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }
        .btn:hover {
            background: #0056b3;
        }
        .info-box {
            background: #e9ecef;
            padding: 15px;
            border-left: 4px solid #007bff;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 NailScanner Control Panel</h1>
            <p>ESP32 Hotspot Interface</p>
        </div>
        
        <div class="info-box">
            <h3>📋 Setup Instructions:</h3>
            <ol>
                <li>Make sure your laptop is connected to "NailScanner" WiFi</li>
                <li>Run your Python app.py on this laptop</li>
                <li>The app should be accessible at: <code>http://192.168.4.2:5000</code></li>
                <li>Mobile users can access: <a href="/mobile">Mobile Interface</a></li>
            </ol>
        </div>
        
        <div class="video-section">
            <h3>📹 Camera Feed Integration</h3>
            <p>To integrate with your Python app, add this ESP32 endpoint to your Flask app:</p>
            <pre style="background: #f8f9fa; padding: 10px; border-radius: 5px;">
@app.route('/esp32_update', methods=['POST'])
def esp32_update():
    try:
        # Send current prediction to ESP32
        data = {
            "prediction": current_prediction,
            "confidence": current_confidence
        }
        requests.post("http://192.168.4.1/api/update", json=data)
        return "OK"
    except:
        return "Error"
            </pre>
        </div>
        
        <div class="status-panel">
            <h3>📊 Current Status</h3>
            <div id="statusDisplay">
                <p><strong>Prediction:</strong> <span id="prediction">Loading...</span></p>
                <p><strong>Confidence:</strong> <span id="confidence">Loading...</span></p>
                <p><strong>Last Update:</strong> <span id="lastUpdate">Loading...</span></p>
                <p><strong>ESP32 Uptime:</strong> <span id="uptime">Loading...</span></p>
            </div>
        </div>
        
        <div style="text-align: center;">
            <button class="btn" onclick="refreshStatus()">🔄 Refresh Status</button>
            <button class="btn" onclick="window.open('/mobile', '_blank')">📱 Open Mobile View</button>
            <button class="btn" onclick="testConnection()">🧪 Test Connection</button>
        </div>
        
        <div class="info-box">
            <h3>🔗 Network Information:</h3>
            <p><strong>ESP32 IP:</strong> 192.168.4.1</p>
            <p><strong>Laptop IP:</strong> 192.168.4.2 (expected)</p>
            <p><strong>Mobile Access:</strong> <a href="http://192.168.4.1/mobile">http://192.168.4.1/mobile</a></p>
        </div>
    </div>

    <script>
        function refreshStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('prediction').textContent = data.prediction;
                    document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
                    document.getElementById('lastUpdate').textContent = new Date(data.timestamp).toLocaleTimeString();
                    document.getElementById('uptime').textContent = data.uptime + ' seconds';
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('prediction').textContent = 'Connection Error';
                });
        }
        
        function testConnection() {
            fetch('/api/update', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    prediction: 'Test from laptop',
                    confidence: 0.95
                })
            })
            .then(response => response.text())
            .then(data => {
                alert('Test successful: ' + data);
                refreshStatus();
            })
            .catch(error => {
                alert('Test failed: ' + error);
            });
        }
        
        // Auto-refresh every 3 seconds
        setInterval(refreshStatus, 3000);
        
        // Initial load
        refreshStatus();
    </script>
</body>
</html>
)";
}

String generateMobileHTML() {
  return R"(
<!DOCTYPE html>
<html>
<head>
    <title>NailScanner Mobile</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
            overflow-x: hidden;
        }
        
        .header {
            text-align: center;
            padding: 20px 15px;
            background: rgba(0,0,0,0.3);
        }
        
        .header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }
        
        .header p {
            opacity: 0.8;
            font-size: 14px;
        }
        
        .main-content {
            padding: 20px 15px;
        }
        
        .prediction-card {
            background: rgba(0,0,0,0.4);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            margin-bottom: 20px;
            border: 2px solid rgba(255,255,255,0.2);
        }
        
        .prediction-text {
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .confidence-text {
            font-size: 18px;
            opacity: 0.9;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .status-item {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .status-label {
            font-size: 12px;
            opacity: 0.7;
            margin-bottom: 5px;
        }
        
        .status-value {
            font-size: 16px;
            font-weight: bold;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .btn {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 10px;
            font-weight: bold;
            cursor: pointer;
            font-size: 14px;
            color: white;
            background: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.3);
        }
        
        .btn:active {
            transform: scale(0.95);
            background: rgba(255,255,255,0.3);
        }
        
        .video-info {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }
        
        .connection-indicator {
            position: fixed;
            top: 15px;
            right: 15px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #22c55e;
            box-shadow: 0 0 10px rgba(34, 197, 94, 0.6);
            z-index: 1000;
        }
        
        .offline {
            background: #ef4444;
            box-shadow: 0 0 10px rgba(239, 68, 68, 0.6);
        }
        
        .healthy { background: rgba(34, 197, 94, 0.8) !important; }
        .warning { background: rgba(234, 179, 8, 0.8) !important; }
        .danger { background: rgba(239, 68, 68, 0.8) !important; }
        
        .footer-info {
            background: rgba(0,0,0,0.2);
            padding: 15px;
            text-align: center;
            font-size: 12px;
            opacity: 0.8;
        }
        
        @media (orientation: landscape) {
            .main-content {
                max-width: 600px;
                margin: 0 auto;
            }
        }
    </style>
</head>
<body>
    <div class="connection-indicator" id="connectionStatus"></div>
    
    <div class="header">
        <h1>🔬 NailScanner</h1>
        <p>Mobile Monitor</p>
    </div>
    
    <div class="main-content">
        <div class="prediction-card" id="predictionCard">
            <div class="prediction-text" id="predictionText">
                Connecting...
            </div>
            <div class="confidence-text" id="confidenceText">
                Please wait...
            </div>
        </div>
        
        <div class="status-grid">
            <div class="status-item">
                <div class="status-label">Status</div>
                <div class="status-value" id="statusValue">Loading</div>
            </div>
            <div class="status-item">
                <div class="status-label">Last Update</div>
                <div class="status-value" id="updateTime">-</div>
            </div>
            <div class="status-item">
                <div class="status-label">ESP32 Uptime</div>
                <div class="status-value" id="uptimeValue">-</div>
            </div>
            <div class="status-item">
                <div class="status-label">WiFi Signal</div>
                <div class="status-value" id="wifiSignal">Strong</div>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="refreshData()">🔄 Refresh</button>
            <button class="btn" onclick="toggleFullscreen()">📱 Fullscreen</button>
        </div>
        
        <div class="video-info">
            <h4>📹 Camera Feed</h4>
            <p>Camera feed is processed on laptop<br>
            Results are displayed here in real-time</p>
        </div>
        
        <div class="footer-info">
            Connected to ESP32 NailScanner Hotspot<br>
            IP: 192.168.4.1 | Network: NailScanner
        </div>
    </div>

    <script>
        let isOnline = true;
        
        function updateData() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update prediction display
                    document.getElementById('predictionText').textContent = data.prediction;
                    document.getElementById('confidenceText').textContent = 
                        data.confidence ? 'Confidence: ' + (data.confidence * 100).toFixed(1) + '%' : 'No confidence data';
                    
                    // Update status values
                    document.getElementById('statusValue').textContent = data.status;
                    document.getElementById('updateTime').textContent = new Date().toLocaleTimeString();
                    document.getElementById('uptimeValue').textContent = Math.floor(data.uptime / 60) + 'm';
                    
                    // Update card color based on prediction
                    const card = document.getElementById('predictionCard');
                    card.className = 'prediction-card';
                    
                    const prediction = data.prediction.toLowerCase();
                    if (prediction.includes('healthy')) {
                        card.classList.add('healthy');
                    } else if (prediction.includes('place') || prediction.includes('waiting')) {
                        card.classList.add('warning');
                    } else {
                        card.classList.add('danger');
                    }
                    
                    // Update connection status
                    isOnline = true;
                    document.getElementById('connectionStatus').className = 'connection-indicator';
                })
                .catch(error => {
                    console.error('Error:', error);
                    isOnline = false;
                    document.getElementById('connectionStatus').className = 'connection-indicator offline';
                    document.getElementById('predictionText').textContent = 'Connection Lost';
                    document.getElementById('confidenceText').textContent = 'Retrying...';
                });
        }
        
        function refreshData() {
            updateData();
            // Visual feedback
            const btn = event.target;
            const originalText = btn.textContent;
            btn.textContent = '⟳ Refreshing...';
            setTimeout(() => {
                btn.textContent = originalText;
            }, 1000);
        }
        
        function toggleFullscreen() {
            if (document.fullscreenElement) {
                document.exitFullscreen();
            } else {
                document.documentElement.requestFullscreen().catch(err => {
                    alert('Fullscreen not supported on this device');
                });
            }
        }
        
        // Auto-update every 2 seconds
        setInterval(updateData, 2000);
        
        // Initial load
        updateData();
        
        // Handle visibility changes
        document.addEventListener('visibilitychange', function() {
            if (!document.hidden) {
                updateData();
            }
        });
        
        // Prevent zoom on double tap
        let lastTouchEnd = 0;
        document.addEventListener('touchend', function (event) {
            const now = (new Date()).getTime();
            if (now - lastTouchEnd <= 300) {
                event.preventDefault();
            }
            lastTouchEnd = now;
        }, false);
    </script>
</body>
</html>
)";
}
