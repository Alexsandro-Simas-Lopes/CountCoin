# Classe do Detector de Moedas

from ultralytics import YOLO
import cv2 as cv
import os
import time
import datetime
from pathlib import Path
from .base import PipelineEtapa

class DetectorMoedas(PipelineEtapa):
    def __init__(self, model_path, video_path, output_dir, class_values, class_names):
        self.model_path = model_path
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.class_values = class_values
        self.class_names = class_names
        self.model = YOLO(model_path)

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

    def executar(self):
        cap = cv.VideoCapture(self.video_path)
        frame_id = 0
        cv.namedWindow('Detecção de Moedas', cv.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Fim do vídeo ou falha ao ler frame.")
                break

            results = self.model.predict(source=frame, conf=0.9, save=False, verbose=False)[0]
            h, w = frame.shape[:2]
            detections, total_value = self.extrair_deteccoes_confiaveis(results, w, h)

            if detections:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img_name = f"frame_{timestamp}_{frame_id:04d}.jpg"
                cv.imwrite(str(self.images_dir / img_name), frame)
                with open(self.labels_dir / img_name.replace(".jpg", ".txt"), "w") as f:
                    for cls, bx, by, bw_, bh_ in detections:
                        f.write(f"{cls} {bx:.6f} {by:.6f} {bw_:.6f} {bh_:.6f}\n")

            self.mostrar_deteccoes(frame, results, total_value)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            frame_id += 1
            time.sleep(0.02) 

        cap.release()
        cv.destroyAllWindows()

    def extrair_deteccoes_confiaveis(self, results, w, h):
        detections = []
        total_value = 0.0
        for box in results.boxes:
            conf = float(box.conf)
            if conf >= 0.99:
                cls = int(box.cls)
                bx, by, bw, bh = box.xywh[0]
                detections.append((cls, bx / w, by / h, bw / w, bh / h))
                total_value += self.class_values.get(cls, 0.0)
        return detections, total_value

    def mostrar_deteccoes(self, frame, results, total_value):
        for box, conf, cls_id in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.conf.cpu().numpy(),
            results.boxes.cls.cpu().numpy()
        ):
            x1, y1, x2, y2 = box.astype(int)
            label_name = self.class_names.get(int(cls_id), 'desconhecido')
            label = f"{label_name} {conf:.2f}"
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv.putText(frame, f"Total: R$ {total_value:.2f}", (10, 30),
            cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv.imshow('Detecção de Moedas', frame)
