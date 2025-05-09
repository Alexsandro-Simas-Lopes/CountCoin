# Script principal de execução do pipeline de detecção

# Importando as bibliotecas necessárias e os módulos do projeto
from pipeline.detector import DetectorMoedas
from pipeline.treinador import TreinadorPseudoLabels
from pipeline.pipeline_exec import PipelineMoeda

if __name__ == "__main__":
    pipeline = PipelineMoeda()
    pipeline.adicionar_etapa(DetectorMoedas(
        model_path='runs/detect/treinamento_moedas/weights/best.pt',
        video_path='tests/.mp4/coin/1_1.mp4',
        output_dir='pseudo_dataset/train',
        class_values={0:1.00, 1:0.10, 2:0.25, 3:0.05, 4:0.50},
        class_names={0: '1.00 Real', 1: '0.10 Cent', 2: '0.25 Cent', 3: '0.05 Cent', 4: '0.50 Cent'}
    ))
    # pipeline.adicionar_etapa(TreinadorPseudoLabels(
    #     model_path='runs/detect/treinamento_moedas/weights/best.pt',
    #     yaml_path='pseudo_dataset/data.yaml'
    # ))
    pipeline.executar_pipeline()
