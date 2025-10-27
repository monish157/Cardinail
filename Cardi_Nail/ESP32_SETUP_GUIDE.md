# CardiNail ESP32 Mobile Setup Guide

## Step-by-Step Instructions

### 1. Hardware Requirements
- ESP32 Dev Module (not ESP32-CAM)
- USB cable for ESP32
- Your laptop with webcam
- Mobile phone
- WiFi network (laptop and mobile should be on same network)

### 2. Install Arduino IDE and ESP32 Libraries

#### 2.1 Install Arduino IDE
1. Download and install Arduino IDE from https://www.arduino.cc/en/software
2. Open Arduino IDE

#### 2.2 Add ESP32 Board Support
1. Go to File → Preferences
2. In "Additional Board Manager URLs" add:
   ```
   https://dl.espressif.com/dl/package_esp32_index.json
   ```
3. Go to Tools → Board → Boards Manager
4. Search for "ESP32" and install "esp32 by Espressif Systems"

#### 2.3 Install Required Libraries
1. Go to Tools → Manage Libraries
2. Install these libraries:
   - **ArduinoJson** by Benoit Blanchon
   - **WiFi** (should be pre-installed with ESP32)
   - **HTTPClient** (should be pre-installed with ESP32)
   - **WebServer** (should be pre-installed with ESP32)

### 3. Configure Your Network Settings

#### 3.1 Find Your Laptop's IP Address
**On Windows (PowerShell):**
```powershell
ipconfig | findstr IPv4
```
**Example output:** `IPv4 Address. . . . . . . . . . . : 192.168.1.105`

#### 3.2 Update ESP32 Code
1. Open `esp32_code.ino` in Arduino IDE
2. Update these lines at the top:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";           // Your WiFi name
   const char* password = "YOUR_WIFI_PASSWORD";   // Your WiFi password
   String flaskServerIP = "192.168.1.105";       // Your laptop's IP from step 3.1
   ```

### 4. Upload Code to ESP32

#### 4.1 Connect ESP32
1. Connect ESP32 to laptop via USB cable
2. In Arduino IDE:
   - Tools → Board → ESP32 Arduino → ESP32 Dev Module
   - Tools → Port → Select the COM port for your ESP32
   - Tools → Upload Speed → 115200

#### 4.2 Upload the Code
1. Click the Upload button (arrow icon)
2. Wait for "Done uploading" message
3. Open Serial Monitor (Tools → Serial Monitor)
4. Set baud rate to 115200
5. Press ESP32 reset button

#### 4.3 Note ESP32 IP Address
In the Serial Monitor, you'll see:
```
WiFi connected!
ESP32 IP address: 192.168.1.108
ESP32 web server started
Access mobile interface at: http://192.168.1.108
```
**Important:** Note down the ESP32 IP address (e.g., 192.168.1.108)

### 5. Start the Flask Application

#### 5.1 Install Python Dependencies
```powershell
cd "D:\ranjith projects\Cardi_Nail"
pip install -r req.txt
```

#### 5.2 Run the Application
```powershell
python app.py
```

You should see:
```
CLASS_NAMES: ['blue_finger', 'clubbing', 'healthy']
Device: cpu
Starting app on 0.0.0.0 5000
* Running on http://127.0.0.1:5000
* Running on http://192.168.1.105:5000  # This is your laptop's IP
```

### 6. Access on Mobile Device

#### 6.1 Connect Mobile to Same WiFi
Make sure your mobile phone is connected to the same WiFi network as your laptop and ESP32.

#### 6.2 Access the Application
On your mobile phone's browser, go to:
- **Direct Flask access:** `http://192.168.1.105:5000/mobile` (use your laptop's IP)
- **Via ESP32:** `http://192.168.1.108` (use your ESP32's IP from step 4.3)

### 7. Mobile Interface Features

#### 7.1 Portrait Mode Optimization
- Video stream automatically adjusts for portrait viewing
- Touch the video to enter fullscreen mode
- Status overlay shows current analysis

#### 7.2 Real-time Updates
- Prediction updates every 2 seconds
- Connection status indicator (green = connected, red = disconnected)
- Analysis counter tracks number of predictions

#### 7.3 Controls
- **Refresh:** Manually update status
- **Fullscreen:** Toggle fullscreen video view
- **Capture:** Save current analysis (notification only for now)

### 8. Troubleshooting

#### 8.1 ESP32 Won't Connect to WiFi
- Check WiFi credentials in code
- Ensure WiFi network is 2.4GHz (ESP32 doesn't support 5GHz)
- Try resetting ESP32

#### 8.2 Mobile Can't Access the Stream
- Ensure all devices are on same WiFi network
- Check if Windows Firewall is blocking port 5000
- Try accessing laptop IP directly: `http://LAPTOP_IP:5000/mobile`

#### 8.3 Video Stream Not Loading
- Check if camera is being used by another application
- Try different camera index in `app.py` (change `CAMERA_INDEX = 0` to `CAMERA_INDEX = 1`)

#### 8.4 ESP32 Serial Monitor Shows Errors
- Check if the Flask server is running
- Verify the Flask server IP in ESP32 code matches your laptop's IP
- Ensure port 5000 is not blocked

### 9. Usage Instructions

#### 9.1 For Nail Analysis
1. Start Flask app on laptop
2. Power on ESP32 (should auto-connect to WiFi)
3. Open mobile browser to ESP32 IP address
4. Position finger in front of laptop camera
5. View real-time analysis on mobile in portrait mode

#### 9.2 Understanding Results
- **"Place your finger"** - White background detected or low confidence
- **"healthy"** - Normal nail detected
- **"blue_finger"** - Cyanosis detected
- **"clubbing"** - Nail clubbing detected

### 10. Network Diagram
```
[Laptop with Camera] ←→ [WiFi Router] ←→ [ESP32 Dev Module] ←→ [Mobile Phone]
    (Flask Server)                        (Web Interface)      (Display)
    192.168.1.105:5000                   192.168.1.108        Browser
```

### 11. Optional Enhancements

#### 11.1 Static IP for ESP32
Add to ESP32 code for consistent IP:
```cpp
IPAddress local_IP(192, 168, 1, 200);
IPAddress gateway(192, 168, 1, 1);
IPAddress subnet(255, 255, 0, 0);
WiFi.config(local_IP, gateway, subnet);
```

#### 11.2 Add Authentication
For security, you can add basic authentication to the Flask routes.

#### 11.3 Data Logging
Modify the code to log predictions to a file for later analysis.

---

## Quick Reference

### URLs to Remember:
- **Flask App:** `http://LAPTOP_IP:5000`
- **Mobile Optimized:** `http://LAPTOP_IP:5000/mobile`
- **ESP32 Interface:** `http://ESP32_IP`
- **API Endpoint:** `http://LAPTOP_IP:5000/esp32_status`

### Important Files:
- `app.py` - Main Flask application
- `esp32_code.ino` - Arduino code for ESP32
- `templates/mobile.html` - Mobile-optimized interface
- `req.txt` - Python dependencies

### Default Ports:
- Flask Server: 5000
- ESP32 Web Server: 80
