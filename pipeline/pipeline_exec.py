# Classe PipelineMoeda para gerenciar as etapas do pipeline de conversão de moedas
# Importando a classe PipelineEtapa do módulo base
# e a biblioteca typing para anotações de tipo

from .base import PipelineEtapa

class PipelineMoeda:
    def __init__(self):
        self.etapas = []

    def adicionar_etapa(self, etapa: PipelineEtapa):
        self.etapas.append(etapa)

    def executar_pipeline(self):
        for etapa in self.etapas:
            etapa.executar()
