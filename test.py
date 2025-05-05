import cv2 as cv
import os
from ultralytics import YOLO

# 1) Carrega seu modelo YOLOv8 treinado para detecção de moedas. 
# O caminho 'best.pt' deve ser o arquivo gerado pelo treinamento.
model = YOLO('runs/detect/treinamento_moedas/weights/best.pt')

# 2) Tenta carregar uma imagem. Se falhar 
# (caminho incorreto ou imagem inválida), exibe erro e encerra o programa.

# 1real.jpeg   Precisao: 0.9162172079086304
# 5cent.jpg    Precisao: 0.8960718512535095
# 10cent.jpeg  Precisao: 0.720712423324585
# 25cent.jpg   Precisao: 0.6950568556785583
# 50cent.jpeg  Precisao: 0.8467552661895752

img = cv.imread('imgs/coin2.jpg')
if img is None:
    print("❌ Erro: não foi possível carregar imgs/coin2.jpg")
    exit()

# 3) Faz a predição diretamente, deixando o YOLO redimensionar
results = model.predict(
    source=img,# source=img: usa a imagem carregada (em formato BGR, como o OpenCV fornece).
    conf=0.5,  # conf=0.5: apenas detecções com confiança mínima de 50% serão consideradas.    
    imgsz=640, # imgsz=640: redimensiona para 640x640 antes da detecção.
    save=False  # save=True: salva uma cópia da imagem com as caixas desenhadas em runs/detect/predict.
)

print("results -->", results[0].boxes.cls, results[0].boxes.cls)

# 4) Mostra quantas caixas (moedas) foram detectadas na imagem.
num = len(results[0].boxes)
print(f"🔎 Número de detecções: {num}")

class_names = {
    0: '1 real',
    1: '10 cent',
    2: '25 cent',
    3: '5 cent',
    4: '50 cent'
}

# valor_moeda = {
#     '1 real': 1.00,
#     '5 cent': 0.05,
#     '10 cent': 0.10,
#     '25 cent': 0.25,
#     '50 cent': 0.50
# }

#total_reais = 0.0

# Desenha as caixas
for box, conf, cls in zip(
    results[0].boxes.xyxy.cpu().numpy(),
    results[0].boxes.conf.cpu().numpy(),
    results[0].boxes.cls.cpu().numpy()
):
    print("box >>>", box)
    print("conf >>>", conf)
    print("cls >>>", cls)
    
    x1, y1, x2, y2 = box.astype(int)
    label_name = class_names.get(int(cls), 'desconhecido')
    label = f"{label_name} {conf:f}"
    #total_reais += valor_moeda.get(label_name, 0.0)

    cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.putText(img, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    print("""
          Moeda reconhecida
          Moeda: {0}
          Precisao: {1}
          """.format(label_name, conf))

# Mostra o total
# print(f"✅ Valor total detectado: R$ {total_reais:.2f}")
# cv.putText(img, f"Total: R$ {total_reais:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Nome original da imagem
nome_arquivo = os.path.basename('imgs/coin2.jpg')  # <- usa o nome da imagem
nome_saida = os.path.join('runs/detect/predict', f"det_{nome_arquivo}")  # Ex: imgs/resultado/det_coin2.jpg
# Salva a imagem com as caixas e o valor total
cv.imwrite(nome_saida, img)
print(f"✅ Imagem salva em: {nome_saida}")

cv.imwrite('runs/detect/predict/i.jpg', img)
print("✅ Imagem salva na route: runs/detect/predict/i.jpg")
# 6) Mostra mesmo que não haja caixas
cv.imshow('Deteccoes', img)
cv.waitKey(0)
cv.destroyAllWindows()

