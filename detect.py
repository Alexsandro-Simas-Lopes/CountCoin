from ultralytics import YOLO
import cv2 as cv
import time
import os
from pathlib import Path

# Carregar modelo treinado
model = YOLO('runs/detect/treinamento_moedas/weights/best.pt')

output_dir = Path("pseudo_dataset/train")
images_dir = output_dir / "images"
labels_dir = output_dir / "labels"

# Fonte de vídeo
video_test = 'tests/.mp4/coin/1_1.mp4'
cap = cv.VideoCapture(video_test)  # ou use 0 para webcam
cv.namedWindow('Detecção de Moedas', cv.WINDOW_NORMAL)

# Nomes e valores das classes (em reais)
class_names = {
    0: '1 real',
    1: '10 cent',
    2: '25 cent',
    3: '5 cent',
    4: '50 cent'
}
class_values = {
    0: 1.00,
    1: 0.10,
    2: 0.25,
    3: 0.05,
    4: 0.50
}

# Criar diretórios para salvar dataset
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Fim do vídeo ou falha ao ler frame.")
        break

    clean_frame = frame.copy()

    results = model.predict(source=frame, conf=0.5, save=False, verbose=False)[0]

    h, w = frame.shape[:2]
    high_conf_detections = []
    total_value = 0.0  # Inicializa o total para cada frame

    for box in results.boxes:
        conf = float(box.conf)
        if conf >= 0.9:
            cls = int(box.cls)
            x_center, y_center, width, height = box.xywh[0]
            # Normalizar para formato YOLO
            x_center /= w
            y_center /= h
            width /= w
            height /= h
            high_conf_detections.append((cls, x_center, y_center, width, height))

            # Somar valor
            total_value += class_values.get(cls, 0.0)

    # Salvar se houver detecção confiável
    if high_conf_detections:
        img_name = f"frame_{frame_id:04d}.jpg"
        img_path = os.path.join(images_dir, img_name)
        cv.imwrite(img_path, clean_frame)

        label_path = os.path.join(labels_dir, img_name.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            for cls, x, y, w_, h_ in high_conf_detections:
                f.write(f"{cls} {x:.6f} {y:.6f} {w_:.6f} {h_:.6f}\n")

    # Exibir com bounding boxes e texto
    for box, conf, cls_id in zip(
        results.boxes.xyxy.cpu().numpy(),
        results.boxes.conf.cpu().numpy(),
        results.boxes.cls.cpu().numpy()
    ):
        x1, y1, x2, y2 = box.astype(int)
        label_name = class_names.get(int(cls_id), 'desconhecido')
        label = f"{label_name} {conf:.2f}"

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Exibir total acumulado
    cv.putText(frame, f"Total: R$ {total_value:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv.imshow('Detecção de Moedas', frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    time.sleep(0.05)
    frame_id += 1

cap.release()
cv.destroyAllWindows()

# Cria arquivo data.yaml temporário
with open("pseudo_dataset/data.yaml", "w") as f:
    f.write("""
path: pseudo_dataset
train: ../train/images
val: ../valid/images
test: ../test/images # pode mudar para um conjunto de validação real
nc: 5  # número de classes
names: ['1 real', '10 cent', '25 cent', '5 cent', '50 cent']  # nomes das classes
""")

# Re-treinar modelo com pseudo-labels
print("\n🚀 Iniciando re-treino com pseudo-labels...")
model.train(data=str("pseudo_dataset/data.yaml"), epochs=30, imgsz=640)
print("✅ Re-treino concluído!")