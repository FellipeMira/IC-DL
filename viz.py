import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import random

# Configurar logging
logging.basicConfig(filename='patch_visualization_log.txt', level=logging.INFO)

def visualize_patches(patch_dir, num_patches=6, channel=0, patch_size=512, colormap='gray', save_path='patches_visualization.png', fig_size=(10, 8), dpi=100, normalize=True):
    """
    Visualiza patches SAR de uma pasta em uma grade de subplots.

    Args:
        patch_dir (str): Diretório com patches .npy (ex.: 'data/train/images').
        num_patches (int): Número de patches a visualizar (ex.: 6 para grade 2x3).
        channel (int): Índice do canal a exibir (ex.: 0 para 0_VV).
        patch_size (int): Tamanho do patch (ex.: 512).
        colormap (str): Mapa de cores para visualização (ex.: 'gray').
        save_path (str): Caminho para salvar a figura.
        fig_size (tuple): Tamanho da figura em polegadas (ex.: (10, 8)).
        dpi (int): Resolução da figura salva.
        normalize (bool): Aplica normalização logarítmica (np.log1p) para SAR.
    """
    # Verificar diretório
    if not os.path.exists(patch_dir):
        logging.error(f'Diretório {patch_dir} não encontrado.')
        raise FileNotFoundError(f'Diretório {patch_dir} não encontrado.')

    # Listar patches .npy
    patch_files = [f for f in os.listdir(patch_dir) if f.endswith('.npy')]
    if not patch_files:
        logging.error(f'Nenhum arquivo .npy encontrado em {patch_dir}.')
        raise ValueError(f'Nenhum arquivo .npy encontrado em {patch_dir}.')

    # Selecionar patches (aleatoriamente)
    selected_patches = random.sample(patch_files, min(num_patches, len(patch_files)))

    # Configurar grade de subplots (ex.: 2x3 para 6 patches)
    rows = (num_patches + 2) // 3  # Ajusta linhas para número de patches
    cols = min(num_patches, 3)
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    axes = axes.flatten() if num_patches > 1 else [axes]

    # Visualizar patches
    for i, patch_file in enumerate(selected_patches):
        try:
            patch_path = os.path.join(patch_dir, patch_file)
            patch = np.load(patch_path)  # Shape: [H, W, C] (ex.: [512, 512, 8])

            # Verificar dimensões
            if patch.shape[:2] != (patch_size, patch_size):
                logging.warning(f'Patch {patch_file} tem tamanho inesperado: {patch.shape}.')
                continue
            if channel >= patch.shape[2]:
                logging.warning(f'Canal {channel} inválido para patch {patch_file} com {patch.shape[2]} canais.')
                continue

            # Selecionar canal e normalizar (se necessário)
            image = patch[:, :, channel]
            if normalize:
                image = np.log1p(np.clip(image, 0, None))  # Normalização log para SAR

            # Plotar
            axes[i].imshow(image, cmap=colormap)
            axes[i].set_title(f'{patch_file}\nCanal {channel} (ex.: 0_VV)', fontsize=8)
            axes[i].axis('off')

        except Exception as e:
            logging.error(f'Erro ao carregar/visualizar {patch_file}: {str(e)}')
            axes[i].set_title(f'Erro: {patch_file}', fontsize=8)
            axes[i].axis('off')

    # Desativar eixos vazios (se houver)
    for i in range(len(selected_patches), len(axes)):
        axes[i].axis('off')

    # Ajustar layout e salvar/exibir
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    logging.info(f'Visualização de {len(selected_patches)} patches salva em {save_path}.')

if __name__ == '__main__':
    # Exemplo de uso
    patch_dir = 'data/train/images'  # Ajuste conforme sua estrutura
    visualize_patches(
        patch_dir=patch_dir,
        num_patches=6,
        channel=0,  # Visualiza canal 0 (ex.: 0_VV)
        patch_size=512,
        colormap='gray',
        save_path='patches_visualization.png',
        fig_size=(10, 8),
        dpi=100,
        normalize=True
    )
