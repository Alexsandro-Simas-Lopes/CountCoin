import pandas as pd
import matplotlib.pyplot as plt

def plot_yolo_results(csv_path):
    # Carrega o arquivo CSV
    df = pd.read_csv(csv_path)
    epochs = df['epoch'] + 1  # começa do 1

    plt.figure(figsize=(14, 6))

    # Box Loss: treino e validação
    plt.subplot(1, 2, 1)
    plt.plot(epochs, df['train/box_loss'], label='Box Loss Treino', marker='o')
    plt.plot(epochs, df['val/box_loss'], label='Box Loss Validação', marker='o')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.title('Box Loss - Treino vs Validação')
    plt.legend()
    plt.grid(True)

    # Class Loss + mAP
    plt.subplot(1, 2, 2)
    plt.plot(epochs, df['train/cls_loss'], label='Class Loss Treino', linestyle='--', marker='x')
    plt.plot(epochs, df['val/cls_loss'], label='Class Loss Validação', linestyle='-', marker='x')
    plt.plot(epochs, df['metrics/mAP50(B)'], label='mAP@0.5', linestyle='-', marker='s')
    plt.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', linestyle=':', marker='s')
    plt.xlabel('Épocas')
    plt.title('Class Loss + mAP')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_yolo_results('runs/detect/treinamento_moedas/results.csv')
