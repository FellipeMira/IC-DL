#!/usr/bin/env python3
"""
Script simples para aplicar o modelo SAR treinado na imagem original completa
Aplica o modelo diretamente sem divisão em patches para evitar efeitos de borda
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from enhanced_sar_config import EnhancedSARAutoencoder, IN_CHANNELS, DEVICE, TEMPORAL_STEPS
from scipy.ndimage import zoom
import os

def load_model():
    """Carrega o modelo treinado"""
    print("📦 Carregando modelo...")
    model = EnhancedSARAutoencoder(in_channels=IN_CHANNELS, temporal_steps=TEMPORAL_STEPS)
    
    checkpoint = torch.load('best_sar_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).float()
    model.eval()
    print("✅ Modelo carregado!")
    return model

def normalize_patch(patch):
    """Normaliza patch SAR de forma robusta"""
    h, w, c = patch.shape
    patch = patch.reshape(h, w, TEMPORAL_STEPS, 2)  # 32 tempos x 2 bandas
    
    for band in range(2):
        band_data = patch[:, :, :, band]  # [H, W, temporal]
        
        # Verificar valores válidos
        valid_mask = ~np.isnan(band_data) & ~np.isinf(band_data)
        
        if np.sum(valid_mask) > 0:  # Se há valores válidos
            if band == 0:  # VV
                # Usar percentis para clipping mais robusto
                p2, p98 = np.percentile(band_data[valid_mask], [2, 98])
                band_data = np.clip(band_data, p2, p98)
                # Normalizar para [0, 1]
                band_data = (band_data - p2) / (p98 - p2 + 1e-8)
            else:  # VH
                # Usar percentis para clipping mais robusto  
                p2, p98 = np.percentile(band_data[valid_mask], [2, 98])
                band_data = np.clip(band_data, p2, p98)
                # Normalizar para [0, 1]
                band_data = (band_data - p2) / (p98 - p2 + 1e-8)
        else:
            # Se não há valores válidos, usar valores padrão
            band_data = np.full_like(band_data, 0.5)  # Valor neutro
            
        patch[:, :, :, band] = band_data
    
    # Verificação final
    result = patch.reshape(h, w, c).astype(np.float32)
    
    # Garantir que não há valores inválidos no resultado
    result = np.nan_to_num(result, nan=0.5, posinf=1.0, neginf=0.0)
    
    return result

def process_image():
    """Processa a imagem inteira diretamente sem patches"""
    print("🛰️ Aplicando modelo na imagem SAR completa...")
    
    # Carregar modelo
    model = load_model()
    
    # Carregar imagem
    image_path = 'data/raw_images/Sentinel_1_ROI_32.tif'
    print(f"📖 Carregando {image_path}...")
    
    with rasterio.open(image_path) as src:
        image = src.read()
        image = np.transpose(image, (1, 2, 0))  # [H, W, C]
        
        # Verificar e tratar dados inválidos
        print(f"   • Valores NaN na imagem original: {np.isnan(image).sum()}")
        print(f"   • Valores Inf na imagem original: {np.isinf(image).sum()}")
        print(f"   • Range original: [{np.nanmin(image):.3f}, {np.nanmax(image):.3f}]")
        
        # Tratar valores inválidos
        if np.isnan(image).any() or np.isinf(image).any():
            print("🔧 Corrigindo valores inválidos na imagem...")
            # Substituir NaN e Inf por valores médios por banda
            for c in range(image.shape[2]):
                band = image[:, :, c]
                if np.isnan(band).any() or np.isinf(band).any():
                    # Usar mediana como valor de preenchimento (mais robusto que média)
                    valid_values = band[~np.isnan(band) & ~np.isinf(band)]
                    if len(valid_values) > 0:
                        fill_value = np.median(valid_values)
                    else:
                        fill_value = -20.0  # Valor típico para SAR
                    
                    band = np.where(np.isnan(band) | np.isinf(band), fill_value, band)
                    image[:, :, c] = band
            
            print(f"   • Range após correção: [{image.min():.3f}, {image.max():.3f}]")
        
        # Ajustar canais se necessário
        if image.shape[2] != IN_CHANNELS:
            if image.shape[2] < IN_CHANNELS:
                repeats = IN_CHANNELS // image.shape[2]
                image = np.repeat(image, repeats, axis=2)
            else:
                image = image[:, :, :IN_CHANNELS]
    
    print(f"✅ Imagem carregada: {image.shape}")
    
    # Redimensionar para dimensões compatíveis com a arquitetura (múltiplos de 16)
    h_orig, w_orig = image.shape[0], image.shape[1]
    
    # Calcular novas dimensões (próximas das originais, mas múltiplas de 16)
    h_new = ((h_orig // 16) + 1) * 16 if h_orig % 16 != 0 else h_orig
    w_new = ((w_orig // 16) + 1) * 16 if w_orig % 16 != 0 else w_orig
    
    print(f"🔧 Redimensionando de {h_orig}x{w_orig} para {h_new}x{w_new} (compatível com U-Net)")
    
    # Redimensionar usando interpolação bilinear
    zoom_factors = (h_new/h_orig, w_new/w_orig, 1)  # Não redimensionar canais
    image_resized = zoom(image, zoom_factors, order=1)  # order=1 = bilinear
    
    print(f"✅ Imagem redimensionada: {image_resized.shape}")
    
    # Normalizar a imagem inteira
    print("🔧 Normalizando imagem completa...")
    print(f"   • Range antes da normalização: [{image_resized.min():.3f}, {image_resized.max():.3f}]")
    image_normalized = normalize_patch(image_resized)
    print(f"   • Range após normalização: [{image_normalized.min():.3f}, {image_normalized.max():.3f}]")
    print(f"   • Contém NaN? {np.isnan(image_normalized).any()}")
    print(f"   • Contém Inf? {np.isinf(image_normalized).any()}")
    
    # Converter para tensor [1, C, H, W]
    print("🧠 Aplicando modelo na imagem completa...")
    image_tensor = torch.from_numpy(image_normalized).float()
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H_new, W_new]
    image_tensor = image_tensor.to(DEVICE)
    
    # Aplicar modelo diretamente na imagem inteira
    with torch.no_grad():
        print("⚡ Processamento direto - sem patches!")
        reconstruction = model(image_tensor)
        
        # Calcular erro de reconstrução pixel por pixel
        error = torch.nn.functional.mse_loss(reconstruction, image_tensor, reduction='none')
        error_map_resized = error.mean(dim=1).squeeze(0).cpu().numpy()  # [H_new, W_new]
        
        print(f"   • Reconstrução shape: {reconstruction.shape}")
        print(f"   • Input tensor shape: {image_tensor.shape}")
        print(f"   • Range reconstrução: [{reconstruction.min().item():.6f}, {reconstruction.max().item():.6f}]")
        print(f"   • Range input tensor: [{image_tensor.min().item():.6f}, {image_tensor.max().item():.6f}]")
        print(f"   • Range erro: [{error_map_resized.min():.6f}, {error_map_resized.max():.6f}]")
        print(f"   • Erro contém NaN? {np.isnan(error_map_resized).any()}")
        print(f"   • Erro contém Inf? {np.isinf(error_map_resized).any()}")
    
    print(f"✅ Erro calculado: {error_map_resized.shape}")
    
    # Redimensionar mapa de erro de volta ao tamanho original
    print("🔙 Redimensionando resultado para tamanho original...")
    zoom_factors_back = (h_orig/h_new, w_orig/w_new)
    error_map = zoom(error_map_resized, zoom_factors_back, order=1)
    
    print(f"✅ Erro de reconstrução final: {error_map.shape}")
    print(f"   • Range erro final: [{error_map.min():.6f}, {error_map.max():.6f}]")
    print(f"   • Erro final contém NaN? {np.isnan(error_map).any()}")
    print(f"   • Erro final contém Inf? {np.isinf(error_map).any()}")
    
    # Tratar NaN e Inf se existirem
    if np.isnan(error_map).any() or np.isinf(error_map).any():
        print("⚠️  Corrigindo valores NaN/Inf no mapa de erro...")
        error_map = np.nan_to_num(error_map, nan=0.0, posinf=1.0, neginf=0.0)
        print(f"   • Range após correção: [{error_map.min():.6f}, {error_map.max():.6f}]")
    
    # Criar mapa de mudanças
    print("🎯 Criando mapa de mudanças...")
    
    # Verificar se temos variação no erro
    if error_map.min() == error_map.max():
        print("⚠️  Erro uniforme detectado! Usando threshold fixo...")
        threshold = error_map.mean() + 0.001  # Threshold mínimo
    else:
        threshold = np.percentile(error_map, 95)  # Top 5% como mudanças
    
    changes = error_map > threshold
    print(f"   • Threshold usado: {threshold:.6f}")
    print(f"   • Pixels com mudança: {np.sum(changes)} / {changes.size}")
    print(f"   • Porcentagem de mudanças: {np.sum(changes)/changes.size*100:.2f}%")
    
    # Salvar resultados
    print("💾 Salvando resultados...")
    
    # Visualização
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Imagem original
    original_vis = np.mean(image[:, :, :8], axis=2)
    original_vis = (original_vis - original_vis.min()) / (original_vis.max() - original_vis.min())
    axes[0].imshow(original_vis, cmap='gray')
    axes[0].set_title('Imagem SAR Original')
    axes[0].axis('off')
    
    # Mapa de erro
    axes[1].imshow(error_map, cmap='hot')
    axes[1].set_title('Erro de Reconstrução')
    axes[1].axis('off')
    
    # Mudanças detectadas
    axes[2].imshow(original_vis, cmap='gray', alpha=0.7)
    axes[2].imshow(changes, cmap='Reds', alpha=0.6)
    axes[2].set_title('Mudanças Detectadas')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('mudancas_detectadas.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Salvar resultados como arquivos GeoTIFF com CRS e extent originais
    print("🗺️ Salvando resultados georreferenciados...")
    
    # Reabrir imagem original para extrair metadados geoespaciais
    with rasterio.open(image_path) as src:
        # Copiar metadados da imagem original
        out_meta = src.meta.copy()
        out_meta.update({
            'count': 1,  # Uma banda por arquivo
            'dtype': 'float32',
            'compress': 'lzw'  # Compressão para economizar espaço
        })
        
        # 1. Salvar mapa de erro de reconstrução
        with rasterio.open('erro_reconstrucao.tif', 'w', **out_meta) as dst:
            dst.write(error_map.astype(np.float32), 1)
            dst.set_band_description(1, 'Erro de Reconstrução SAR')
        
        # 2. Salvar mapa binário de mudanças
        out_meta_binary = out_meta.copy()
        out_meta_binary.update({'dtype': 'uint8'})
        
        with rasterio.open('mudancas_binario.tif', 'w', **out_meta_binary) as dst:
            dst.write(changes.astype(np.uint8) * 255, 1)  # 0 = sem mudança, 255 = mudança
            dst.set_band_description(1, 'Mudanças Detectadas (255=mudança)')
        
        # 3. Salvar imagem original visualizada para comparação
        with rasterio.open('imagem_original_vis.tif', 'w', **out_meta) as dst:
            dst.write(original_vis.astype(np.float32), 1)
            dst.set_band_description(1, 'Imagem SAR Original (Visualização)')
    
    print("✅ Arquivos GeoTIFF salvos:")
    print("   📁 erro_reconstrucao.tif - Mapa de erro contínuo")
    print("   📁 mudancas_binario.tif - Mudanças binárias (0/255)")
    print("   📁 imagem_original_vis.tif - Imagem original para comparação")
    
    # Estatísticas
    change_percent = np.sum(changes) / changes.size * 100
    print(f"\n📊 RESULTADOS:")
    print(f"   • Erro médio: {error_map.mean():.6f}")
    print(f"   • Threshold: {threshold:.6f}")
    print(f"   • Área com mudanças: {change_percent:.2f}%")
    print(f"   • Visualização PNG: mudancas_detectadas.png")
    print(f"   • Dados georreferenciados: *.tif")
    print(f"   • Processamento: IMAGEM COMPLETA (sem patches!)")
    
    return error_map, changes

if __name__ == '__main__':
    if not os.path.exists('best_sar_model.pth'):
        print("❌ Modelo não encontrado! Execute o treinamento primeiro.")
    elif not os.path.exists('data/raw_images/Sentinel_1_ROI_32.tif'):
        print("❌ Imagem não encontrada!")
    else:
        process_image()
