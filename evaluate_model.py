#!/usr/bin/env python3
"""
Script para avaliar e visualizar os resultados da rede neural autoencoder SAR
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from enhanced_sar_config import EnhancedSARAutoencoder, TemporalSARDataset, enhanced_val_transforms, IN_CHANNELS, DEVICE, TEMPORAL_STEPS
import os

def load_trained_model(model_path='best_sar_model.pth'):
    """Carrega o modelo treinado"""
    model = EnhancedSARAutoencoder(
        in_channels=IN_CHANNELS,
        temporal_steps=TEMPORAL_STEPS
    )
    
    # Carregar checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).float()  # CORREÇÃO: Garantir que modelo está no dispositivo correto
    model.eval()
    return model

def evaluate_model(model, test_dataset):
    """Avalia o modelo no conjunto de teste"""
    model.eval()
    total_loss = 0
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            if len(sample.shape) == 3:
                sample = sample.unsqueeze(0)  # Adicionar batch dimension
            
            sample = sample.to(DEVICE).float()  # CORREÇÃO: Garantir float32 no dispositivo correto
            reconstruction = model(sample)
            loss = criterion(reconstruction, sample)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_dataset)
    return avg_loss

def visualize_reconstructions(model, test_dataset, num_samples=3):
    """Visualiza reconstruções do modelo"""
    model.eval()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            # Carregar amostra
            sample = test_dataset[i]
            if len(sample.shape) == 3:
                sample_batch = sample.unsqueeze(0)
            else:
                sample_batch = sample
                
            sample_batch = sample_batch.to(DEVICE).float()  # CORREÇÃO: Garantir float32
            reconstruction = model(sample_batch)
            
            # Converter para numpy para visualização
            original = sample.cpu().numpy()
            reconstructed = reconstruction.squeeze(0).cpu().numpy()
            
            # Calcular média dos canais para visualização (64 canais -> 1 canal)
            if len(original.shape) == 3:
                original_vis = np.mean(original, axis=0)  # Média dos canais
                reconstructed_vis = np.mean(reconstructed, axis=0)
            else:
                original_vis = original
                reconstructed_vis = reconstructed
            
            # Plotar original
            axes[0, i].imshow(original_vis, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Plotar reconstrução
            axes[1, i].imshow(reconstructed_vis, cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f'Reconstrução {i+1}')
            axes[1, i].axis('off')
            
            # Calcular MSE para esta amostra
            mse = np.mean((original - reconstructed) ** 2)
            print(f"Amostra {i+1} - MSE: {mse:.6f}")
    
    plt.tight_layout()
    plt.savefig('sar_autoencoder_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Visualização salva como 'sar_autoencoder_results.png'")

def analyze_feature_maps(model, test_sample):
    """Analisa os mapas de características do encoder"""
    model.eval()
    
    if len(test_sample.shape) == 3:
        test_sample = test_sample.unsqueeze(0)
    
    test_sample = test_sample.to(DEVICE).float()  # CORREÇÃO: Garantir float32
    
    # Extrair features de cada camada do encoder
    with torch.no_grad():
        # Encoder features
        e1 = model.enc1(test_sample)
        e2 = model.enc2(e1)
        e3 = model.enc3(e2)
        latent = model.enc4(e3)
        
        print(f"Input shape: {test_sample.shape}")
        print(f"Encoder 1 shape: {e1.shape}")
        print(f"Encoder 2 shape: {e2.shape}")
        print(f"Encoder 3 shape: {e3.shape}")
        print(f"Latent space shape: {latent.shape}")
        
        # Visualizar algumas features do espaço latente
        latent_np = latent.squeeze(0).cpu().numpy()
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i in range(8):
            row = i // 4
            col = i % 4
            if i < latent_np.shape[0]:
                axes[row, col].imshow(latent_np[i], cmap='viridis')
                axes[row, col].set_title(f'Feature Map {i+1}')
                axes[row, col].axis('off')
            else:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('latent_features.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Mapas de características salvos como 'latent_features.png'")

def calculate_metrics(model, test_dataset):
    """Calcula métricas detalhadas"""
    model.eval()
    
    mse_scores = []
    psnr_scores = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            if len(sample.shape) == 3:
                sample_batch = sample.unsqueeze(0)
            else:
                sample_batch = sample
                
            sample_batch = sample_batch.to(DEVICE).float()  # CORREÇÃO: Garantir float32
            reconstruction = model(sample_batch)
            
            # Converter para numpy
            original = sample.cpu().numpy()
            reconstructed = reconstruction.squeeze(0).cpu().numpy()
            
            # MSE
            mse = np.mean((original - reconstructed) ** 2)
            mse_scores.append(mse)
            
            # PSNR (Peak Signal-to-Noise Ratio)
            if mse > 0:
                max_pixel = 1.0  # Nossos dados estão normalizados para [0,1]
                psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
                psnr_scores.append(psnr)
    
    return {
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores),
        'psnr_mean': np.mean(psnr_scores),
        'psnr_std': np.std(psnr_scores),
        'mse_scores': mse_scores,
        'psnr_scores': psnr_scores
    }

def main():
    """Função principal para avaliar o modelo"""
    print("=== AVALIAÇÃO DA REDE NEURAL AUTOENCODER SAR ===")
    print("=" * 50)
    
    # Verificar se o modelo existe
    if not os.path.exists('best_sar_model.pth'):
        print("❌ Modelo treinado não encontrado! Execute o treinamento primeiro.")
        return
    
    # Carregar modelo
    print("📦 Carregando modelo treinado...")
    model = load_trained_model()
    print("✓ Modelo carregado com sucesso!")
    
    # Carregar conjunto de teste
    print("📁 Carregando conjunto de teste...")
    test_dataset = TemporalSARDataset('data/test/images', transforms=enhanced_val_transforms, temporal_augment=False)
    print(f"✓ Conjunto de teste carregado: {len(test_dataset)} amostras")
    
    # Avaliar modelo
    print("🔍 Avaliando modelo...")
    test_loss = evaluate_model(model, test_dataset)
    print(f"✓ Loss médio no teste: {test_loss:.6f}")
    
    # Calcular métricas detalhadas
    print("📊 Calculando métricas detalhadas...")
    metrics = calculate_metrics(model, test_dataset)
    print(f"✓ MSE: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
    print(f"✓ PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
    
    # Visualizar reconstruções
    print("🖼️  Gerando visualizações...")
    visualize_reconstructions(model, test_dataset, num_samples=min(3, len(test_dataset)))
    
    # Analisar espaço latente
    print("🧠 Analisando espaço latente...")
    if len(test_dataset) > 0:
        analyze_feature_maps(model, test_dataset[0])
    
    # Resumo final
    print("\n=== RESUMO DOS RESULTADOS ===")
    print(f"📈 Perda de teste: {test_loss:.6f}")
    print(f"📊 MSE médio: {metrics['mse_mean']:.6f}")
    print(f"📡 PSNR médio: {metrics['psnr_mean']:.2f} dB")
    print(f"🎯 Número de amostras testadas: {len(test_dataset)}")
    
    # Interpretação dos resultados
    print("\n=== INTERPRETAÇÃO ===")
    if test_loss < 0.1:
        print("🟢 Excelente: A rede conseguiu reconstruir as imagens SAR com baixo erro")
    elif test_loss < 0.2:
        print("🟡 Bom: A rede apresenta reconstrução satisfatória")
    else:
        print("🔴 Necessita melhoria: Considere ajustar hiperparâmetros ou arquitetura")
    
    if metrics['psnr_mean'] > 20:
        print("🟢 PSNR indica boa qualidade de reconstrução")
    elif metrics['psnr_mean'] > 15:
        print("🟡 PSNR indica qualidade moderada")
    else:
        print("🔴 PSNR baixo - considere melhorias na arquitetura")

if __name__ == '__main__':
    main()
