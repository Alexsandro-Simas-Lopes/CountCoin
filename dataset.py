import os
import random
import shutil

# Caminho para o diretório do seu dataset
dataset_dir = 'datasets'

# Proporções para treino, validação e teste
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Caminhos para as pastas de imagens e labels
images_dir = os.path.join(dataset_dir, 'train/images')
labels_dir = os.path.join(dataset_dir, 'train/labels')

# Listar todas as imagens
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_files)

# Calcular quantidades para cada conjunto
total_images = len(image_files)
train_count = int(total_images * train_ratio)
val_count = int(total_images * val_ratio)

# Dividir as imagens
train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]

# Função para mover arquivos
def move_files(file_list, subset):
    subset_images_dir = os.path.join(dataset_dir, subset, 'images')
    subset_labels_dir = os.path.join(dataset_dir, subset, 'labels')
    os.makedirs(subset_images_dir, exist_ok=True)
    os.makedirs(subset_labels_dir, exist_ok=True)
    for file_name in file_list:
        # Mover imagem
        src_image = os.path.join(images_dir, file_name)
        dst_image = os.path.join(subset_images_dir, file_name)
        shutil.move(src_image, dst_image)
        # Mover label correspondente
        label_name = os.path.splitext(file_name)[0] + '.txt'
        src_label = os.path.join(labels_dir, label_name)
        dst_label = os.path.join(subset_labels_dir, label_name)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)

# Mover os arquivos para as respectivas pastas
move_files(train_files, 'train')
move_files(val_files, 'valid')
move_files(test_files, 'test')

print('Divisão do dataset concluída com sucesso!')
