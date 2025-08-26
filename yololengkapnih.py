import cv2
import time
import json
import numpy as np
import paho.mqtt.client as mqtt 
from gpiozero import LED
from ultralytics import YOLO

MQTT_BROKER = "192.168.0.174"
MQTT_PORT = 1883
MQTT_USERNAME = "cpsmagang"
MQTT_PASSWORD = "cpsjaya123"
DEVICE_IP_ADDRESS = "192.168.0.174" 
SENSOR_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/sensor"
ACTION_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/action"

# inisialisasi kamera model gpio pin
model_pose = YOLO("yolo11n-pose_ncnn_model", task="pose") 

cam_source = "usb0"
resW, resH = 640, 480
gpio_pin = 26

led = LED(gpio_pin)

if "usb" in cam_source:
    cam_type = "usb"
    cam_idx = int(cam_source[3:])
    cam = cv2.VideoCapture(cam_idx)
    cam.set(3, resW)
    cam.set(4, resH)
    if not cam.isOpened():
        print("Gagal membuka kamera.")
        exit()
else:
    print("ga ada kamera")
    exit()
print("Kamera siap.")

consecutive_detections = 0
gpio_state = 0 # 0 = OFF, 1 = ON. 
is_person_reported = False 
fps_buffer = []
fps_avg_len = 50

def control_device(device, action):
    "Fungsi yang dipanggil SAAT ADA PERINTAH dari backend."
    global gpio_state
    if device == "lamp":
        if action == "turn_on":
            led.on()
            gpio_state = 1
        elif action == "turn_off":
            led.off()
            gpio_state = 0
        print(f"🚀 AKSI DARI BACKEND: Menjalankan '{action}' pada '{device}'")

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"✅ Terhubung ke MQTT Broker di {MQTT_BROKER}!")
        client.subscribe(ACTION_TOPIC)
        print(f"👂 SUBSCRIBE ke topik aksi: {ACTION_TOPIC}")
    else:
        print(f"❌ Gagal terhubung ke MQTT, kode error: {rc}")

def on_message(client, userdata, msg):
    print(f"📩 PESAN DITERIMA di topik {msg.topic}")
    try:
        payload = json.loads(msg.payload.decode())
        device = payload.get("device")
        action = payload.get("action")
        if device and action:
            control_device(device, action)
    except Exception as e:
        print(f"Error memproses pesan: {e}")



# --- Inisialisasi MQTT Client ---
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

print("\n🚀 Sistem deteksi mulai berjalan. Tekan 'Q' untuk berhenti.")
try:
    while True:
        t_start = time.perf_counter()

        # Pengambilan frame (dari kode Anda)
        ret, frame = cam.read()
        if not ret:
            print("Peringatan: Gagal mengambil frame.")
            break

        # Deteksi pose (dari kode Anda)
        results = model_pose.predict(frame, verbose=False)
        annotated_frame = results[0].plot()
        pose_found = len(results) > 0 and len(results[0].keypoints) > 0

        # Logika smoothing (dari kode Anda)
        if pose_found:
            consecutive_detections = min(consecutive_detections + 1, 10)
        else:
            consecutive_detections = max(consecutive_detections - 1, 0)

        
        # Tentukan apakah status deteksi harus aktif atau tidak
        should_be_active = consecutive_detections >= 8 and gpio_state == 0
        should_be_inactive = consecutive_detections <= 0 and gpio_state == 1

        if should_be_active:
            gpio_state = 1
            led.on()
        elif should_be_inactive:
            gpio_state = 0
            led.off()

        # Kirim laporan ke backend HANYA jika status berubah
        if should_be_active and not is_person_reported:
            is_person_reported = True
            payload = json.dumps({"motion_detected": True})
            client.publish(SENSOR_TOPIC, payload) # <-- Melaporkan ke backend
            print(f" PUBLISH: Pose Terdeteksi!")

        elif should_be_inactive and is_person_reported:
            is_person_reported = False
            payload = json.dumps({"motion_cleared": True})
            client.publish(SENSOR_TOPIC, payload) # <-- Melaporkan ke backend
            print(f"📡 PUBLISH: Pose Tidak Terdeteksi!")

        # Menampilkan visualisasi (dari kode Anda)
        if gpio_state == 0:
            cv2.putText(annotated_frame, "Light OFF", (20,60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255), 2)
        else:
            cv2.putText(annotated_frame, "Light ON", (20,60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)

        t_stop = time.perf_counter()
        if (t_stop - t_start) > 0:
            frame_rate_calc = 1 / (t_stop - t_start)
            fps_buffer.append(frame_rate_calc)
            if len(fps_buffer) > fps_avg_len:
                fps_buffer.pop(0)
            avg_frame_rate = np.mean(fps_buffer)
            cv2.putText(annotated_frame, f'FPS: {avg_frame_rate:.2f}', (20,30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
        
        cv2.imshow("Smart Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # --- Cleanup ---
    print("\n🧹 Membersihkan sumber daya...")
    cam.release()
    cv2.destroyAllWindows()
    led.close() # <-- Membersihkan pin GPIO
    client.loop_stop()
    client.disconnect()
    print("Selesai.")
