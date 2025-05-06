import cv2 as cv
from ultralytics import YOLO

model = YOLO('runs/detect/treinamento_moedas/weights/best.pt')

# Use 0 para webcam, ou substitua por caminho de vídeo (ex: 'video.mp4')
video_path = 0  # ou 'videos/moedas.mp4'
cap = cv.VideoCapture(video_path)

class_names = {
    0: '1 real',
    1: '10 cent',
    2: '25 cent',
    3: '5 cent',
    4: '50 cent'
}
# valor_moeda = {'1 real': 1.00,'5 cent': 0.05,'10 cent': 0.10,'25 cent': 0.25,'50 cent': 0.50}

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Fim do vídeo ou falha ao ler frame.")
        break

    # Realiza predição no frame atual - model.track()
    results = model.predict(
        source=frame,
        conf=0.85,
        imgsz=416,
        save=False,
        verbose=False
    )

    # Filtra as melhores detecções por classe
    melhores_deteccoes = {}

    for box, conf, cls_id in zip(
        results[0].boxes.xyxy.cpu().numpy(),
        results[0].boxes.conf.cpu().numpy(),
        results[0].boxes.cls.cpu().numpy()
    ):
        if cls_id not in melhores_deteccoes or conf > melhores_deteccoes[cls_id]['conf']:
            melhores_deteccoes[cls_id] = {'conf': conf, 'box': box}

    # Desenha as detecções no frame
    for cls_id, info in melhores_deteccoes.items():
        box = info['box']
        conf = info['conf']
        x1, y1, x2, y2 = box.astype(int)
        label_name = class_names.get(int(cls_id), 'desconhecido')
        label = f"{label_name} {conf:.2f}"

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Exibe o frame com detecções
    cv.imshow('Detecção de Moedas', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


