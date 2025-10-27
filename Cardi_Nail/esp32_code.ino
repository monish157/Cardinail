// esp32_nail_ap.ino
#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>

WebServer server(80);

const char* ap_ssid = "NailScanner";
const char* ap_pass = "12345678";

// The static IP you will set on the PC when it connects to this AP:
const char* pc_ip = "192.168.4.2";
const int pc_port = 5000;

String pc_api_output_url() {
  return String("http://") + pc_ip + ":" + String(pc_port) + "/api/output";
}
String pc_video_feed_url() {
  return String("http://") + pc_ip + ":" + String(pc_port) + "/video_feed";
}

void handleRoot() {
  String html = "<html><body style='text-align:center;font-family:Arial'>";
  html += "<h2>ESP32 Nail Gateway</h2>";
  html += "<p><a href='/output_page'>Output (proxied)</a></p>";
  html += "<p><a href='/live_page'>Live (embedded)</a></p>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

void handleOutputPage() {
  String page = R"rawliteral(
    <!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"></head><body style="text-align:center;font-family:Arial">
    <h2>Latest Prediction (proxied)</h2><pre id="out">Loading...</pre>
    <script>
    async function f(){try{let r = await fetch('/get_output'); let j = await r.json(); document.getElementById('out').innerText = JSON.stringify(j, null, 2);}catch(e){document.getElementById('out').innerText = 'Error: '+e;}}
    f(); setInterval(f,1000);
    </script></body></html>
  )rawliteral";
  server.send(200, "text/html", page);
}

void handleGetOutput() {
  HTTPClient http;
  String url = pc_api_output_url();
  http.begin(url);
  int code = http.GET();
  String payload = "{}";
  if (code > 0) payload = http.getString();
  else payload = String("{\"error\":\"http code=") + String(code) + "\"}";
  http.end();
  server.send(200, "application/json", payload);
}

void handleLivePage() {
  String vid = pc_video_feed_url();
  String page = "<html><body style='text-align:center;font-family:Arial'><h2>Live Camera</h2>"
                "<p>If blank, ensure PC is connected to this AP and its IP is " + String(pc_ip) + ".</p>"
                "<img src='" + vid + "' style='max-width:100%;height:auto;border:1px solid #ccc'>"
                "</body></html>";
  server.send(200, "text/html", page);
}

void setup() {
  Serial.begin(115200);
  delay(300);
  WiFi.softAP(ap_ssid, ap_pass);
  Serial.print("AP started at ");
  Serial.println(WiFi.softAPIP());
  server.on("/", handleRoot);
  server.on("/output_page", handleOutputPage);
  server.on("/get_output", handleGetOutput);
  server.on("/live_page", handleLivePage);
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
}