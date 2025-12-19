import cv2
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import threading
from queue import Queue

VIDEO_SOURCE = "http://10.49.117.187:4747/video"

# Ekran döndürme ayarı: 0=normal, 1=90°, 2=180°, 3=270°
rotation_mode = 1
fullscreen = False

# Hız dönüşüm faktörü: piksel/saniye -> km/saat
# Bu değer kameranın yüksekliği ve açısına göre kalibre edilmeli
# Varsayılan: 1 piksel = 0.01 metre (yaklaşık değer, gerçek kullanımda kalibre edilmeli)
PIXEL_TO_METER = 0.01  # Metre/piksel
METER_TO_KMH = 3.6     # m/s -> km/h dönüşüm faktörü

model = YOLO("yolov8n-seg.pt")
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Ağ gecikmesini azaltmak için agresif optimizasyonlar
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer'ı küçült (eski frame'leri atla)
cap.set(cv2.CAP_PROP_FPS, 30)  # FPS limiti
# Çözünürlüğü düşür - performans için kritik
# Not: Kamera 640x480 gönderir, 90° döndürme sonrası 480x640 olur
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(2)
if not cap.isOpened():
    print("❌ Kamera açılamadı")
    exit()

print("✅ Kamera açıldı")
print("📱 Klavye Kontrolleri:")
print("   'r' tuşu: Ekranı döndür (0° → 90° → 180° → 270° → 0°)")
print("   'f' tuşu: Full ekran aç/kapat")
print("   'q' tuşu: Çıkış")
print("\n💡 Ağ Gecikmesi İpuçları:")
print("   - WiFi yerine kablolu bağlantı kullanın")
print("   - Telefon ve bilgisayar aynı WiFi ağında olmalı")
print("   - DroidCam'de düşük kalite modunu deneyin")
print("   - Router'a yakın olun")

# Pencereyi oluştur
cv2.namedWindow("INSAN HIZ TESPITI", cv2.WINDOW_NORMAL)

# Threading için frame queue
frame_queue = Queue(maxsize=2)  # Sadece en son 2 frame'i tut
latest_frame = None
frame_lock = threading.Lock()

# Her insan için önceki pozisyon ve zaman bilgilerini sakla
person_tracks = defaultdict(lambda: {'prev_center': None, 'prev_time': None, 'id': None})
next_id = 0

# FPS hesaplama için
fps_start_time = time.time()
fps_frame_count = 0
fps = 0

# Frame okuma thread fonksiyonu
def read_frames():
    """Frame'leri ayrı thread'de oku"""
    global latest_frame
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()
        else:
            time.sleep(0.01)  # Hata durumunda kısa bekleme

# Frame okuma thread'ini başlat
frame_thread = threading.Thread(target=read_frames, daemon=True)
frame_thread.start()
time.sleep(1)  # Thread'in başlaması için bekle

def calculate_iou(mask1, mask2):
    """İki mask arasındaki IoU (Intersection over Union) değerini hesapla - optimize edilmiş"""
    # Daha hızlı hesaplama için örnekleme
    h, w = mask1.shape
    step = max(1, min(h, w) // 50)  # Her 50 pikselde bir örnekle
    
    mask1_sampled = mask1[::step, ::step]
    mask2_sampled = mask2[::step, ::step]
    
    intersection = np.logical_and(mask1_sampled, mask2_sampled).sum()
    union = np.logical_or(mask1_sampled, mask2_sampled).sum()
    if union == 0:
        return 0
    return intersection / union

def merge_close_detections(detections, iou_threshold=0.4, distance_threshold=60):
    """Birbirine çok yakın algılamaları birleştir"""
    if len(detections) == 0:
        return []
    
    merged = []
    used = [False] * len(detections)
    
    for i, det1 in enumerate(detections):
        if used[i]:
            continue
        
        # Bu algılamayı birleştirilmiş listeye ekle
        merged_det = det1.copy()
        used[i] = True
        
        # Diğer algılamalarla karşılaştır
        for j, det2 in enumerate(detections):
            if i == j or used[j]:
                continue
            
            # Merkez mesafesi kontrolü
            cx1, cy1 = det1['center']
            cx2, cy2 = det2['center']
            dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            
            # IoU kontrolü
            iou = calculate_iou(det1['mask'], det2['mask'])
            
            # Eğer çok yakınsa birleştir
            if dist < distance_threshold or iou > iou_threshold:
                # Daha büyük mask'ı kullan (daha güvenilir)
                if det2['mask'].sum() > merged_det['mask'].sum():
                    merged_det = det2.copy()
                used[j] = True
        
        merged.append(merged_det)
    
    return merged

while True:
    # Threading'den en son frame'i al
    with frame_lock:
        if latest_frame is None:
            time.sleep(0.01)
            continue
        frame = latest_frame.copy()
    
    # Ekranı döndür
    if rotation_mode == 1:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_mode == 2:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_mode == 3:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    h, w = frame.shape[:2]
    current_time = time.time()
    
    # FPS hesaplama
    fps_frame_count += 1
    if current_time - fps_start_time >= 1.0:
        fps = fps_frame_count / (current_time - fps_start_time)
        fps_frame_count = 0
        fps_start_time = current_time

    # Sadece insanları algıla (class 0 = person)
    # 90° döndürülmüş görüntü için optimize edilmiş: 480x640 boyutlarına uygun imgsz
    # Döndürülmüş görüntü 480x640 olduğu için 480 kullanıyoruz
    results = model(frame, conf=0.4, classes=[0], imgsz=480, verbose=False, half=False)

    # Tüm algılamaları topla
    all_detections = []
    
    for r in results:
        if r.masks is None:
            continue

        for i, mask_data in enumerate(r.masks.data):
            mask = mask_data.cpu().numpy()
            # Mask'ı frame boyutuna resize et
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Daha hızlı merkez hesaplama - sadece örnekleme yap
            ys, xs = np.where(mask > 0.5)
            if len(ys) < 10:  # Çok küçük mask'ları atla
                continue

            # Örnekleme ile daha hızlı hesaplama
            step = max(1, len(ys) // 100)  # Maksimum 100 nokta kullan
            cy = int(np.mean(ys[::step]))
            cx = int(np.mean(xs[::step]))
            
            all_detections.append({
                'center': (cx, cy),
                'mask': mask
            })

    # Çift algılamaları birleştir
    merged_detections = merge_close_detections(all_detections)
    
    # Her birleştirilmiş algılamayı işle
    for det in merged_detections:
        cx, cy = det['center']
        
        # En yakın takip edilen kişiyi bul veya yeni ID ata
        min_dist = float('inf')
        matched_id = None
        
        for person_id, track_data in person_tracks.items():
            if track_data['prev_center'] is not None:
                prev_cx, prev_cy = track_data['prev_center']
                dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                # 90° döndürülmüş görüntü için optimize edilmiş mesafe eşiği (480x640 boyutlarına göre)
                if dist < min_dist and dist < 120:  # 120 piksel mesafe eşiği (dikey görüntü için optimize)
                    min_dist = dist
                    matched_id = person_id

        if matched_id is None:
            matched_id = next_id
            next_id += 1
            person_tracks[matched_id]['id'] = matched_id

        # Hız hesapla
        track_data = person_tracks[matched_id]
        speed = 0.0
        
        if track_data['prev_center'] is not None and track_data['prev_time'] is not None:
            prev_cx, prev_cy = track_data['prev_center']
            dt = current_time - track_data['prev_time']
            
            if dt > 0:
                # Öklid mesafesi kullanarak hız hesapla (piksel/saniye)
                distance_px = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                speed_px_per_s = distance_px / dt
                
                # Piksel/saniye -> metre/saniye -> km/saat
                speed_m_per_s = speed_px_per_s * PIXEL_TO_METER
                speed = speed_m_per_s * METER_TO_KMH  # km/h

        # Güncelle
        person_tracks[matched_id]['prev_center'] = (cx, cy)
        person_tracks[matched_id]['prev_time'] = current_time

        # Görselleştir
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
        cv2.circle(frame, (cx, cy), 12, (0, 255, 0), 2)
        
        # Hız bilgisini göster (km/h)
        speed_text = f"ID:{matched_id} Hiz: {speed:.2f} km/h"
        cv2.putText(frame, speed_text, (cx - 70, cy - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Uzun süre görünmeyen kişileri temizle (5 saniye)
    to_remove = []
    for person_id, track_data in person_tracks.items():
        if track_data['prev_time'] is not None:
            if current_time - track_data['prev_time'] > 5.0:
                to_remove.append(person_id)
    
    for person_id in to_remove:
        del person_tracks[person_id]

    # Frame başına bilgi göster
    info_text = f"Tespit Edilen Insan: {len(merged_detections)}"
    cv2.putText(frame, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # FPS göster
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Döndürme bilgisini göster
    rotation_texts = ["Normal", "90°", "180°", "270°"]
    cv2.putText(frame, f"Ekran: {rotation_texts[rotation_mode]}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("INSAN HIZ TESPITI", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        rotation_mode = (rotation_mode + 1) % 4
        print(f"📱 Ekran döndürme: {rotation_texts[rotation_mode]}")
    elif key == ord('f'):
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty("INSAN HIZ TESPITI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("📺 Full ekran: AÇIK")
        else:
            cv2.setWindowProperty("INSAN HIZ TESPITI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            print("📺 Full ekran: KAPALI")

cap.release()
cv2.destroyAllWindows()