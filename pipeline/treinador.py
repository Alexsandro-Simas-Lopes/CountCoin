# Classe de Treinamento com Pseudo-Labels

from ultralytics import YOLO
from .base import PipelineEtapa

class TreinadorPseudoLabels(PipelineEtapa):
    def __init__(self, model_path, yaml_path):
        self.model = YOLO(model_path)
        self.yaml_path = yaml_path

    def executar(self):
        print("\n🚀 Iniciando re-treino com pseudo-labels...")
        self.model.train(data=self.yaml_path, epochs=30, imgsz=640)
        print("✅ Re-treino finalizado.")
