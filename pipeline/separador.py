import os
import shutil
import random
from pathlib import Path
from .base import PipelineEtapa

class SeparadorDataset(PipelineEtapa):
    def __init__(self, dataset_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        self.dataset_dir = Path(dataset_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def executar(self):
        images_dir = self.dataset_dir / 'images'
        labels_dir = self.dataset_dir / 'labels'

        all_images = sorted(list(images_dir.glob("*.jpg")))
        random.shuffle(all_images)

        total = len(all_images)
        train_cutoff = int(total * self.train_ratio)
        val_cutoff = train_cutoff + int(total * self.val_ratio)

        datasets = {
            'train': all_images[:train_cutoff],
            'val': all_images[train_cutoff:val_cutoff],
            'test': all_images[val_cutoff:]
        }

        for split, images in datasets.items():
            for kind in ['images', 'labels']:
                target_dir = self.dataset_dir.parent / split / kind
                os.makedirs(target_dir, exist_ok=True)

            for img_path in images:
                label_path = labels_dir / img_path.name.replace('.jpg', '.txt')
                shutil.move(str(img_path), str(self.dataset_dir.parent / split / 'images' / img_path.name))
                shutil.move(str(label_path), str(self.dataset_dir.parent / split / 'labels' / label_path.name))

        # Remover diretórios antigos
        shutil.rmtree(images_dir, ignore_errors=True)
        shutil.rmtree(labels_dir, ignore_errors=True)
        print("✅ Dataset separado em train, val e test.")
