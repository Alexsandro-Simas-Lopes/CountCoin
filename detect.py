# Script principal de execução do pipeline de detecção

# Importando as bibliotecas necessárias e os módulos do projeto
from pipeline.detector import DetectorMoedas
from pipeline.treinador import TreinadorPseudoLabels
from pipeline.pipeline_exec import PipelineMoeda
from pipeline.separador import SeparadorDataset  # ✅ novo import

if __name__ == "__main__":
    pipeline = PipelineMoeda()
    pipeline.adicionar_etapa(DetectorMoedas(
        model_path='runs/detect/treinamento_moedas/weights/best.pt',
        video_path='tests/.mp4/9.mp4',
        output_dir='pseudo_dataset/temp',  # temp = pasta temporária
        class_values={0:1.00, 1:0.10, 2:0.25, 3:0.05, 4:0.50},
        class_names={0: '1.00 Real', 1: '0.10 Cent', 2: '0.25 Cent', 3: '0.05 Cent', 4: '0.50 Cent'}
    ))
    pipeline.adicionar_etapa(SeparadorDataset(
        dataset_dir='pseudo_dataset/temp',  # separa o conteúdo gerado
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    ))
    # pipeline.adicionar_etapa(TreinadorPseudoLabels(
    #     model_path='runs/detect/treinamento_moedas/weights/best.pt',
    #     yaml_path='pseudo_dataset/data.yaml'
    # ))
    pipeline.executar_pipeline()

