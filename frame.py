import cv2 as cv
import time
import os
from ultralytics import YOLO

from ultralytics.utils.ops import non_max_suppression

model = YOLO('runs/detect/treinamento_moedas/weights/best.pt')
web_cam = 0
video_test = 'test/9.mp4'
cap = cv.VideoCapture(video_test)
cv.namedWindow('Detecção de Moedas', cv.WINDOW_NORMAL)

class_names = {
    0: '1 real',
    1: '10 cent',
    2: '25 cent',
    3: '5 cent',
    4: '50 cent'
}

# Pastas
os.makedirs('train/images', exist_ok=True)
os.makedirs('train/labels', exist_ok=True)

contador = 177

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Fim do vídeo ou falha ao ler frame.")
        break

    results = model.predict(source=frame, conf=0.5, imgsz=416, save=False, verbose=False)
    h, w, _ = frame.shape

    # Desenhar todas as caixas detectadas
    for box, conf, cls_id in zip(
        results[0].boxes.xyxy.cpu().numpy(),
        results[0].boxes.conf.cpu().numpy(),
        results[0].boxes.cls.cpu().numpy()
    ):
        x1, y1, x2, y2 = box.astype(int)
        label_name = class_names.get(int(cls_id), 'desconhecido')
        label = f"{label_name} {conf:.2f}"

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv.imshow('Detecção de Moedas', frame)
    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s'):
        image_name = f"datasets/train/images/frame_{contador}.jpg"
        label_name = f"datasets/train/labels/frame_{contador}.txt"
        cv.imwrite(image_name, frame)
        print(f"💾 Imagem salva: {image_name}")

        with open(label_name, 'w') as f:
            for box, conf, cls_id in zip(
                results[0].boxes.xyxy.cpu().numpy(),
                results[0].boxes.conf.cpu().numpy(),
                results[0].boxes.cls.cpu().numpy()
            ):
                x1, y1, x2, y2 = box
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                f.write(f"{int(cls_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        print(f"📝 Label salva: {label_name}")
        contador += 1

    time.sleep(0.05)
