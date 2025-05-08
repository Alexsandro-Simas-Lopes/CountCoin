# Script completo para autoaprendizado com YOLOv8
# Detecta objetos com alta confianca, salva imagens e cria pseudo-labels

from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# Caminhos principais
video_path = "./test/.mp4/4.mp4"  # seu vídeo
model_path = "./yolov8n.pt"  # ou o modelo já treinado
output_dir = Path("pseudo_dataset")
images_dir = output_dir / "images"
labels_dir = output_dir / "labels"
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# Carrega modelo YOLO
model = YOLO(model_path)

# Configuração
CONFIDENCE_THRESHOLD = 0.9
FRAME_SKIP = 5  # processa 1 a cada 5 frames

# Leitura de vídeo
cap = cv2.VideoCapture(video_path)
frame_id = 0
saved_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % FRAME_SKIP == 0:
        results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        for r in results:
            if len(r.boxes) > 0:
                image_name = f"frame_{saved_id:04}.jpg"
                label_name = f"frame_{saved_id:04}.txt"

                # Salva imagem
                cv2.imwrite(str(images_dir / image_name), frame)

                # Cria arquivo de label
                with open(labels_dir / label_name, "w") as f:
                    for box in r.boxes:
                        cls = int(box.cls.item())
                        conf = float(box.conf.item())
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        # Converte para formato YOLO (cx, cy, w, h) normalizado
                        h, w = frame.shape[:2]
                        cx = ((x1 + x2) / 2) / w
                        cy = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h

                        f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                saved_id += 1

    frame_id += 1

cap.release()
print(f"✅ {saved_id} imagens salvas com pseudo-labels em {output_dir}")

# --- Etapa extra: Re-treinar modelo automaticamente com as pseudo-labels ---

# Cria arquivo data.yaml temporário
with open(output_dir / "data.yaml", "w") as f:
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
model.train(data=str(output_dir / "data.yaml"), epochs=30, imgsz=640)
print("✅ Re-treino concluído!")
