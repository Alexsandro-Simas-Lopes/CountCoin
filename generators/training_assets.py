from ultralytics import YOLO

# 1. Escolher o modelo base (ex: yolov8n.pt)
model = YOLO('yolov8n.pt')

# 2. Treinar o modelo com parâmetros extras
model.train(
    data='data/data.yaml',     # Caminho para o data.yaml
    epochs=100,                    # Número total de épocas
    imgsz=640,                     # Tamanho da imagem
    batch=16,                      # Tamanho do batch
    name='treinamento_moedas',    # Nome do projeto (vai criar uma pasta runs/detect/treinamento_moedas)

    patience=10,                   # Early stopping: para se não melhorar em 10 épocas
    save=True,                     # Salva pesos em todas as épocas (útil para análise)
    save_period=5,                 # Salva pesos a cada 5 épocas
    val=True,                      # Ativa validação a cada época
    verbose=True,                  # Mostra logs detalhados
    plots=True                     # Gera gráficos de loss, precisão etc.
)