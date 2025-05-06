import cv2 as cv
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = YOLO('runs/detect/treinamento_moedas/weights/best.pt')
gif_path = 'gifs/1real.gif'
nome_arquivo = os.path.basename(gif_path)

class_names = {
    0: '1 real',
    1: '10 cent',
    2: '25 cent',
    3: '5 cent',
    4: '50 cent'
}
valor_moeda = {
    '1 real': 1.00, '5 cent': 0.05, '10 cent': 0.10,
    '25 cent': 0.25, '50 cent': 0.50
}

# Lê os frames do GIF com PIL
gif = Image.open(gif_path)
frames = []
try:
    while True:
        frame = gif.convert('RGB')
        frame = np.array(frame)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        frame = cv.resize(frame, (640, 640))
        frames.append(frame)
        gif.seek(gif.tell() + 1)
except EOFError:
    pass

print(f"📽️ Total de frames lidos: {len(frames)}")

# Processa os frames
for idx, frame in enumerate(frames):
    results = model.predict(
        source=frame,
        conf=0.80,
        imgsz=640,
        save=False
    )

    total_reais = 0.0
    melhores_deteccoes = {}

    for box, conf, cls_id in zip(
        results[0].boxes.xyxy.cpu().numpy(),
        results[0].boxes.conf.cpu().numpy(),
        results[0].boxes.cls.cpu().numpy()
    ):
        if cls_id not in melhores_deteccoes or conf > melhores_deteccoes[cls_id]['conf']:
            melhores_deteccoes[cls_id] = {'conf': conf, 'box': box}

    for cls_id, info in melhores_deteccoes.items():
        box = info['box']
        conf = info['conf']
        x1, y1, x2, y2 = box.astype(int)

        label_name = class_names.get(int(cls_id), 'desconhecido')
        label = f"{label_name} {conf:.2f}"
        total_reais += valor_moeda.get(label_name, 0.0)

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        print(f"[Frame {idx}] Moeda: {label_name}, Precisão: {conf:.2f}")

    # Escreve total no frame
    cv.putText(frame, f"Total: R$ {total_reais:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Salva cada frame como imagem (opcional)
    cv.imwrite(f'runs/detect/predict/frame_{idx:03}.jpg', frame)

    # Exibe frame
    cv.imshow('Deteccao de Moedas - GIF', frame)
    if cv.waitKey(200) & 0xFF == ord('q'):  # espera 200ms entre os frames
        break

cv.destroyAllWindows()
