import os
import numpy as np
import rasterio
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

# Configurações
PATCH_SIZE = 128  # Ou 256/512 para patches menores
OVERLAP = 0.3  # 30% de sobreposição
STRIDE = int(PATCH_SIZE * (1 - OVERLAP))  # Stride = patch_size/2 para 50% overlap
INPUT_DIR = 'data/raw_images'  # Pasta com imagens SAR originais (.tif)
OUTPUT_DIR = 'data'  # Pasta base para salvar patches
SPLIT_RATIOS = {'train': 0.7, 'val': 0.15, 'test': 0.15}  # Proporções para divisão
SEED = 42  # Para reprodutibilidade

# Configurar logging
logging.basicConfig(filename='patch_creation_log.txt', level=logging.INFO)

def create_patches(image_path, patch_size, stride, output_dir, split, normalize=True):
    """
    Cria patches a partir de uma imagem SAR e salva em output_dir/split/images.
    
    Args:
        image_path (str): Caminho para a imagem SAR (.tif).
        patch_size (int): Tamanho do patch (ex.: 512).
        stride (int): Passo da janela deslizante.
        output_dir (str): Diretório base para salvar patches.
        split (str): 'train', 'val' ou 'test'.
        normalize (bool): Aplica normalização logarítmica (np.log1p).
    """
    # Criar diretórios de saída
    split_dir = os.path.join(output_dir, split, 'images')
    os.makedirs(split_dir, exist_ok=True)
    
    # Ler imagem com rasterio
    with rasterio.open(image_path) as src:
        image = src.read()  # Shape: [C, H, W] (ex.: [8, H, W] para 4 tempos x 2 bandas)
        image = np.transpose(image, (1, 2, 0))  # Para [H, W, C]
        height, width, channels = image.shape
    
    # Normalizar (opcional, para SAR)
    if normalize:
        image = np.log1p(np.clip(image, 0, None))  # Evita valores negativos
    
    # Extrair patches
    patch_count = 0
    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size, :]  # [patch_size, patch_size, C]
            
            # Ignorar patches com muitos valores inválidos (ex.: NaN)
            if np.any(np.isnan(patch)) or np.sum(patch == 0) > 0.5 * patch.size:
                continue
            
            # Salvar patch como .npy
            patch_path = os.path.join(split_dir, f'patch_{i}_{j}_{split}.npy')
            np.save(patch_path, patch)
            patch_count += 1
    
    logging.info(f'Imagem {image_path}: {patch_count} patches criados para {split}.')
    return patch_count

def split_and_patch_images(input_dir, output_dir, patch_size, stride, split_ratios, seed):
    """
    Divide imagens em train/val/test e cria patches para cada conjunto.
    
    Args:
        input_dir (str): Pasta com imagens SAR originais.
        output_dir (str): Pasta base para salvar patches.
        patch_size (int): Tamanho do patch.
        stride (int): Passo da janela deslizante.
        split_ratios (dict): Proporções para train/val/test.
        seed (int): Semente para divisão aleatória.
    """
    # Listar imagens
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    if not image_files:
        raise ValueError(f'Nenhuma imagem .tif encontrada em {input_dir}')
    
    # Dividir em train/val/test
    train_files, temp_files = train_test_split(image_files, train_size=split_ratios['train'], random_state=seed)
    val_files, test_files = train_test_split(temp_files, train_size=split_ratios['val']/(split_ratios['val'] + split_ratios['test']), random_state=seed)
    
    # Processar cada conjunto
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        total_patches = 0
        for file in files:
            image_path = os.path.join(input_dir, file)
            total_patches += create_patches(image_path, patch_size, stride, output_dir, split)
        logging.info(f'Total de {total_patches} patches criados para {split}.')

if __name__ == '__main__':
    # Executar divisão e criação de patches
    split_and_patch_images(INPUT_DIR, OUTPUT_DIR, PATCH_SIZE, STRIDE, SPLIT_RATIOS, SEED)
    print('Criação de patches concluída. Verifique o log em patch_creation_log.txt.')
