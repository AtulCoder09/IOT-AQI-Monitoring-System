import asyncio
import json
import logging
from typing import Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# In a real app, you would load a pre-trained scikit-learn model here:
# import joblib
# model = joblib.load("aqi_anomaly_model.pkl")

app = FastAPI(title="AQI Intelligence Layer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AQI-Backend")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        # Separate the Edge Device (ESP32) and the Dashboard (Next.js) connections
        self.edge_device: WebSocket = None
        self.dashboards: List[WebSocket] = []

    async def connect_edge(self, websocket: WebSocket):
        await websocket.accept()
        self.edge_device = websocket
        logger.info("Edge Device Connected")

    def disconnect_edge(self, websocket: WebSocket):
        self.edge_device = None
        logger.info("Edge Device Disconnected")

    async def connect_dashboard(self, websocket: WebSocket):
        await websocket.accept()
        self.dashboards.append(websocket)
        logger.info(f"Dashboard Connected. Total: {len(self.dashboards)}")

    def disconnect_dashboard(self, websocket: WebSocket):
        self.dashboards.remove(websocket)
        logger.info("Dashboard Disconnected")

    async def broadcast_to_dashboards(self, message: str):
        for connection in self.dashboards:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending to dashboard: {e}")

    async def send_command_to_edge(self, command: dict):
        if self.edge_device:
            try:
                await self.edge_device.send_json(command)
                logger.info(f"Command sent to Edge: {command}")
            except Exception as e:
                logger.error(f"Error sending to Edge: {e}")
        else:
            logger.warning("Edge device not connected. Cannot send command.")

manager = ConnectionManager()

def convert_to_ppm(data: dict) -> dict:
    """
    Convert raw analog sensor readings (0-4095 for ESP32 12-bit ADC)
    to approximate PPM values using sensor characteristic curves.
    """
    mq135_raw = data.get("mq135", 0)
    mq8_raw = data.get("mq8", 0)
    mq9_raw = data.get("mq9", 0)
    dust_raw = data.get("dust", 0)

    # MQ-135: CO2/NH3/Benzene — typical range 10-1000 ppm
    # Rs/Ro ratio approximation from datasheet curve
    mq135_ratio = max(mq135_raw / 4095.0, 0.001)
    mq135_ppm = round(10.0 * pow(mq135_ratio * 3.6, 2.1), 1)

    # MQ-8: Hydrogen — typical range 100-10000 ppm
    mq8_ratio = max(mq8_raw / 4095.0, 0.001)
    mq8_ppm = round(100.0 * pow(mq8_ratio * 3.0, 1.8), 1)

    # MQ-9: CO/Combustible gas — typical range 10-1000 ppm
    mq9_ratio = max(mq9_raw / 4095.0, 0.001)
    mq9_ppm = round(10.0 * pow(mq9_ratio * 4.0, 2.0), 1)

    # Dust/PM2.5: Convert analog voltage to µg/m³ (displayed as PPM equivalent)
    dust_voltage = (dust_raw / 4095.0) * 3.3
    dust_ppm = round(max(0.17 * dust_voltage * 1000 - 0.1, 0), 1)

    return {
        "mq135_ppm": mq135_ppm,
        "mq8_ppm": mq8_ppm,
        "mq9_ppm": mq9_ppm,
        "dust_ppm": dust_ppm
    }

def predict_anomaly(data: dict) -> dict:
    """
    Simulated ML Prediction Layer (Scikit-Learn).
    In production, this would use a model.predict([features]) call.
    """
    mq135 = data.get("mq135", 0)
    mq8 = data.get("mq8", 0)
    mq9 = data.get("mq9", 0)
    dust = data.get("dust", 0)
    temperature = data.get("temperature", 0.0)
    humidity = data.get("humidity", 0.0)

    # Simple heuristic to simulate an ML anomaly detection
    # High MQ9 (Combustible gas) or High Dust/Smoke
    score = (mq135 * 0.1) + (mq8 * 0.2) + (mq9 * 0.5) + (dust * 0.2)
    
    # ---- Total AQI Calculation ----
    # Weighted sub-index formula based on analog readings
    # Each sensor contributes proportionally to overall air quality
    # Scaled to 0-500 range (EPA AQI standard)
    sub_mq135 = min((mq135 / 4095.0) * 500, 500)  # Air quality gas
    sub_mq9   = min((mq9 / 4095.0) * 500, 500)     # CO / Combustible
    sub_mq8   = min((mq8 / 4095.0) * 500, 500)     # Hydrogen
    sub_dust  = min((dust / 4095.0) * 500, 500)     # PM2.5 / Dust
    
    # Weighted average: CO is most dangerous, then air quality, then dust, then H2
    total_aqi = (sub_mq135 * 0.30) + (sub_mq9 * 0.35) + (sub_dust * 0.20) + (sub_mq8 * 0.15)
    total_aqi = round(min(total_aqi, 500), 1)
    
    # AQI Category
    if total_aqi <= 50:
        aqi_category = "Good"
    elif total_aqi <= 100:
        aqi_category = "Moderate"
    elif total_aqi <= 150:
        aqi_category = "Unhealthy (Sensitive)"
    elif total_aqi <= 200:
        aqi_category = "Unhealthy"
    elif total_aqi <= 300:
        aqi_category = "Very Unhealthy"
    else:
        aqi_category = "Hazardous"
    
    anomaly = False
    fan_speed = 0
    status = "Normal"

    if score > 500:
        anomaly = True
        status = "CRITICAL: Hazard Detected"
        fan_speed = 255
    elif score > 200:
        anomaly = True
        status = "WARNING: Poor Air Quality"
        fan_speed = 128

    return {
        "anomaly": anomaly,
        "status": status,
        "risk_score": round(score, 2),
        "recommended_fan_speed": fan_speed,
        "total_aqi": total_aqi,
        "aqi_category": aqi_category,
        "temperature": round(temperature, 1),
        "humidity": round(humidity, 1)
    }

@app.websocket("/ws/edge-node")
async def websocket_edge_endpoint(websocket: WebSocket):
    await manager.connect_edge(websocket)
    try:
        while True:
            # Receive data from ESP32
            data_str = await websocket.receive_text()
            try:
                data = json.loads(data_str)
                logger.info(f"Received Sensor Data: {data}")
                
                # --- INTELLIGENCE LAYER ---
                # Run Scikit-Learn logic (simulated here)
                prediction = predict_anomaly(data)
                
                # --- ACT LAYER ---
                # Check if we need to adjust the fan automatically
                # (Closed-Loop Feedback System)
                # We compare recommended speed with the edge node's current reported speed.
                if prediction["recommended_fan_speed"] != data.get("fan_speed", 0):
                    command = {"fan_speed": prediction["recommended_fan_speed"]}
                    await manager.send_command_to_edge(command)

                # --- PPM CONVERSION LAYER ---
                ppm_values = convert_to_ppm(data)

                # --- INSIGHT LAYER ---
                # Forward data + prediction to connected Next.js dashboards
                payload = {
                    "type": "telemetry",
                    "sensors": data,
                    "ppm": ppm_values,
                    "ml_insights": prediction
                }
                await manager.broadcast_to_dashboards(json.dumps(payload))

            except json.JSONDecodeError:
                logger.error("Invalid JSON from Edge Node")
    except WebSocketDisconnect:
        manager.disconnect_edge(websocket)

@app.websocket("/ws/dashboard")
async def websocket_dashboard_endpoint(websocket: WebSocket):
    await manager.connect_dashboard(websocket)
    try:
        while True:
            # Listen for manual commands from the Next.js Dashboard
            data_str = await websocket.receive_text()
            data = json.loads(data_str)
            
            logger.info(f"Received Command from Dashboard: {data}")

            if data.get("command") == "calibrate":
                logger.info("Triggering Calibration Mode on Edge!")
                await manager.send_command_to_edge({"command": "calibrate"})
                
            elif data.get("command") == "manual_fan":
                speed = int(data.get("speed", 0))
                await manager.send_command_to_edge({"fan_speed": speed})

    except WebSocketDisconnect:
        manager.disconnect_dashboard(websocket)

@app.get("/")
def read_root():
    return {"status": "Antigravity API is running offline/online"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
