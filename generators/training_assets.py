from ultralytics import YOLO

# 1. Escolha o modelo base (ex: yolov8n.pt)
model = YOLO('yolov8n.pt')  

# 2. Treinar o modelo no seu dataset
model.train(
    data='data/data.yaml',  # <-- o seu data.yaml exportado
    epochs=50,                      # número de épocas de treinamento
    imgsz=640,                      # tamanho das imagens
    batch=16,                       # tamanho do batch
    name='treinamento_moedas'        # nome da pasta onde vai salvar os resultados
)