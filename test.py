import cv2 as cv
import os
from ultralytics import YOLO

# Precisao:-------02/05-16:30----------05/05-11:30
# 1.real = 0.9162172079086304 | 0.9661579132080078
# 05cent = 0.8960718512535095 | 0.8534362316131592
# 10cent = 0.720712423324585  | 0.9661579132080078
# 25cent = 0.6950568556785583 | 0.94121915102005  
# 50cent = 0.8467552661895752 | 0.9661579132080078

model = YOLO('runs/detect/treinamento_moedas/weights/best.pt')
url_img = 'test/imgs/6.jpg' # Caminho da imagem
nome_arquivo = os.path.basename(url_img) # Nome do Arquivo

# Função de pré-processamento
def preprocessar_imagem(caminho, tamanho=(640, 640)):
    img = cv.imread(caminho)
    if img is None:
        raise ValueError(f"Erro ao carregar imagem: {caminho}")
    img = cv.resize(img, tamanho)  # YOLO aceita várias resoluções, default é 640x640
    # img = cv.GaussianBlur(img, (3, 3), 0)  Suavização leve
    return img 

# Pré-processamento da imagem
img = preprocessar_imagem(url_img)
# img = cv.imread(url_img)

# Predição com YOLO (imagem já processada)
results = model.predict(
    source=img,
    conf=0.50,    
    imgsz=640,
    save=False
)
print("results -->", results[0].boxes.cls)

# Total detectado
num = len(results[0].boxes)
print(f"🔎 Número de detecções: {num}")

class_names = {
    0: '1 real',
    1: '10 cent',
    2: '25 cent',
    3: '5 cent',
    4: '50 cent'
}
valor_moeda = {
    '1 real': 1.00, '5 cent': 0.05, '10 cent': 0.10,
    '25 cent': 0.25, '50 cent': 0.50
}
total_reais = 0.0
melhores_deteccoes = {}

# Salva a melhor detecção de cada classe
for box, conf, cls_id in zip(
    results[0].boxes.xyxy.cpu().numpy(),
    results[0].boxes.conf.cpu().numpy(),
    results[0].boxes.cls.cpu().numpy()
):
    if cls_id not in melhores_deteccoes or conf > melhores_deteccoes[cls_id]['conf']:
        melhores_deteccoes[cls_id] = {'conf': conf, 'box': box}

# Anotações na imagem
for cls_id, info in melhores_deteccoes.items():
    box = info['box']
    conf = info['conf']
    x1, y1, x2, y2 = box.astype(int)

    label_name = class_names.get(int(cls_id), 'desconhecido')
    label = f"{label_name} {conf:f}"
    total_reais += valor_moeda.get(label_name, 0.0)

    cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.putText(img, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    print(f"""
        Moeda reconhecida
        Moeda: {label_name}
        Precisao: {conf:f}
    """)

# Salvar imagem
cv.putText(img, f"Total: R$ {total_reais:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

nome_saida = os.path.join('runs/detect/predict', f"det_{nome_arquivo}")
cv.imwrite(nome_saida, img)
cv.imwrite('runs/detect/predict/i.jpg', img)

print(f"✅ Imagem salva em: {nome_saida}")
print("✅ Imagem salva na route: runs/detect/predict/i.jpg")

cv.imshow('Deteccoes', img)
cv.waitKey(0)
cv.destroyAllWindows()

