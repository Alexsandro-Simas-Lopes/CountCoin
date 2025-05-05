from PIL import Image
import os

def converter_jpeg_para_png(pasta_origem, pasta_destino):
    """
    Converte todas as imagens .jpeg/.jpg de uma pasta para o formato .png.

    Args:
        pasta_origem (str): Caminho da pasta com as imagens JPEG.
        pasta_destino (str): Caminho da pasta onde as imagens PNG serão salvas.
    """
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    for nome_arquivo in os.listdir(pasta_origem):
        if nome_arquivo.lower().endswith((".jpeg", ".jpg")):
            caminho_imagem = os.path.join(pasta_origem, nome_arquivo)
            with Image.open(caminho_imagem) as img:
                nome_novo = os.path.splitext(nome_arquivo)[0] + ".png"
                caminho_novo = os.path.join(pasta_destino, nome_novo)
                img.save(caminho_novo, "PNG")
                print(f"Convertido: {nome_arquivo} -> {nome_novo}")


converter_jpeg_para_png("50", "50cnt")