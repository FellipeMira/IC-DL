#!/usr/bin/env python3
"""
Módulo para análise e visualização de mudanças temporais em dados SAR
Específico para detecção de enchentes e mudanças de LULC
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from scipy import ndimage
import cv2

class ChangeDetectionAnalyzer:
    """Analisador especializado para detecção de mudanças SAR"""
    
    def __init__(self, model, temporal_steps=32, bands=2):
        self.model = model
        self.temporal_steps = temporal_steps
        self.bands = bands
        self.model.eval()
        
    def compute_reconstruction_error(self, data):
        """Calcula erro de reconstrução pixel a pixel"""
        with torch.no_grad():
            reconstruction = self.model(data)
            error = torch.abs(data - reconstruction)
            return error.cpu().numpy()
    
    def temporal_change_analysis(self, data):
        """Analisa mudanças ao longo do tempo"""
        b, c, h, w = data.shape
        
        # Reshape para análise temporal
        temporal_data = data.view(b, self.temporal_steps, self.bands, h, w)
        
        # Calcular variância temporal para cada banda
        temporal_var = torch.var(temporal_data, dim=1)  # [batch, bands, h, w]
        
        # Detectar mudanças abruptas
        temporal_diff = torch.abs(temporal_data[:, 1:] - temporal_data[:, :-1])
        change_intensity = torch.mean(temporal_diff, dim=1)  # [batch, bands, h, w]
        
        return temporal_var.cpu().numpy(), change_intensity.cpu().numpy()
    
    def flood_detection_features(self, data):
        """Extrai características específicas para detecção de enchentes"""
        b, c, h, w = data.shape
        temporal_data = data.view(b, self.temporal_steps, self.bands, h, w)
        
        # Características específicas para enchente
        features = {}
        
        # 1. Análise de backscatter VV/VH
        vv_data = temporal_data[:, :, 0, :, :]  # Banda VV
        vh_data = temporal_data[:, :, 1, :, :]  # Banda VH
        
        # 2. Razão VV/VH (diminui em áreas alagadas)
        vh_vv_ratio = vh_data / (vv_data + 1e-8)
        features['vh_vv_ratio_mean'] = torch.mean(vh_vv_ratio, dim=1)
        features['vh_vv_ratio_var'] = torch.var(vh_vv_ratio, dim=1)
        
        # 3. Mudança abrupta no backscatter (indicativo de enchente)
        vv_change = torch.abs(vv_data[:, 1:] - vv_data[:, :-1])
        vh_change = torch.abs(vh_data[:, 1:] - vh_data[:, :-1])
        
        features['vv_change_max'] = torch.max(vv_change, dim=1)[0]
        features['vh_change_max'] = torch.max(vh_change, dim=1)[0]
        
        # 4. Coerência temporal (áreas estáveis vs instáveis)
        vv_coherence = 1 / (1 + torch.var(vv_data, dim=1))
        vh_coherence = 1 / (1 + torch.var(vh_data, dim=1))
        
        features['vv_coherence'] = vv_coherence
        features['vh_coherence'] = vh_coherence
        
        return {k: v.cpu().numpy() for k, v in features.items()}
    
    def create_change_map(self, error_map, threshold_percentile=95):
        """Cria mapa binário de mudanças"""
        threshold = np.percentile(error_map, threshold_percentile)
        change_map = (error_map > threshold).astype(np.uint8)
        
        # Pós-processamento morfológico
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        change_map = cv2.morphologyEx(change_map, cv2.MORPH_CLOSE, kernel)
        change_map = cv2.morphologyEx(change_map, cv2.MORPH_OPEN, kernel)
        
        return change_map
    
    def analyze_sample(self, data, sample_name="Sample"):
        """Análise completa de uma amostra"""
        print(f"\n=== Análise de {sample_name} ===")
        
        # Erro de reconstrução
        error = self.compute_reconstruction_error(data)
        mean_error = np.mean(error)
        max_error = np.max(error)
        
        print(f"Erro médio de reconstrução: {mean_error:.6f}")
        print(f"Erro máximo: {max_error:.6f}")
        
        # Análise temporal
        temp_var, change_intensity = self.temporal_change_analysis(data)
        print(f"Variância temporal média: {np.mean(temp_var):.6f}")
        print(f"Intensidade de mudança média: {np.mean(change_intensity):.6f}")
        
        # Características de enchente
        flood_features = self.flood_detection_features(data)
        print(f"Razão VH/VV média: {np.mean(flood_features['vh_vv_ratio_mean']):.3f}")
        print(f"Coerência VV média: {np.mean(flood_features['vv_coherence']):.3f}")
        
        # Mapa de mudanças
        error_mean = np.mean(error[0], axis=0)  # Média dos canais
        change_map = self.create_change_map(error_mean)
        change_percentage = np.sum(change_map) / change_map.size * 100
        print(f"Área com mudanças: {change_percentage:.2f}%")
        
        return {
            'error': error,
            'temporal_variance': temp_var,
            'change_intensity': change_intensity,
            'flood_features': flood_features,
            'change_map': change_map
        }

def visualize_change_analysis(analyzer, data, save_path="change_analysis.png"):
    """Visualiza análise completa de mudanças"""
    
    results = analyzer.analyze_sample(data[0:1], "Test Sample")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 1. Erro de reconstrução (média dos canais)
    error_vis = np.mean(results['error'][0], axis=0)
    im1 = axes[0, 0].imshow(error_vis, cmap='hot')
    axes[0, 0].set_title('Erro de Reconstrução')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. Variância temporal VV
    temp_var_vv = results['temporal_variance'][0, 0]
    im2 = axes[0, 1].imshow(temp_var_vv, cmap='viridis')
    axes[0, 1].set_title('Variância Temporal VV')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. Variância temporal VH
    temp_var_vh = results['temporal_variance'][0, 1]
    im3 = axes[0, 2].imshow(temp_var_vh, cmap='viridis')
    axes[0, 2].set_title('Variância Temporal VH')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 4. Razão VH/VV
    vh_vv_ratio = results['flood_features']['vh_vv_ratio_mean'][0]
    im4 = axes[0, 3].imshow(vh_vv_ratio, cmap='RdYlBu')
    axes[0, 3].set_title('Razão VH/VV')
    axes[0, 3].axis('off')
    plt.colorbar(im4, ax=axes[0, 3])
    
    # 5. Intensidade de mudança VV
    change_vv = results['change_intensity'][0, 0]
    im5 = axes[1, 0].imshow(change_vv, cmap='Reds')
    axes[1, 0].set_title('Mudança VV')
    axes[1, 0].axis('off')
    plt.colorbar(im5, ax=axes[1, 0])
    
    # 6. Intensidade de mudança VH
    change_vh = results['change_intensity'][0, 1]
    im6 = axes[1, 1].imshow(change_vh, cmap='Reds')
    axes[1, 1].set_title('Mudança VH')
    axes[1, 1].axis('off')
    plt.colorbar(im6, ax=axes[1, 1])
    
    # 7. Coerência temporal
    coherence = (results['flood_features']['vv_coherence'][0] + 
                results['flood_features']['vh_coherence'][0]) / 2
    im7 = axes[1, 2].imshow(coherence, cmap='coolwarm')
    axes[1, 2].set_title('Coerência Temporal')
    axes[1, 2].axis('off')
    plt.colorbar(im7, ax=axes[1, 2])
    
    # 8. Mapa de mudanças binário
    im8 = axes[1, 3].imshow(results['change_map'], cmap='gray')
    axes[1, 3].set_title('Mapa de Mudanças')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Análise visual salva em: {save_path}")

def compare_models(original_model, enhanced_model, test_data):
    """Compara performance entre modelo original e melhorado"""
    
    print("\n=== COMPARAÇÃO DE MODELOS ===")
    
    with torch.no_grad():
        # Predições
        orig_pred = original_model(test_data)
        enh_pred = enhanced_model(test_data)
        
        # Erros
        orig_error = torch.mean((test_data - orig_pred) ** 2).item()
        enh_error = torch.mean((test_data - enh_pred) ** 2).item()
        
        print(f"Modelo Original - MSE: {orig_error:.6f}")
        print(f"Modelo Melhorado - MSE: {enh_error:.6f}")
        print(f"Melhoria: {((orig_error - enh_error) / orig_error * 100):.2f}%")
        
        # Análise de detecção de mudanças
        analyzer_orig = ChangeDetectionAnalyzer(original_model)
        analyzer_enh = ChangeDetectionAnalyzer(enhanced_model)
        
        results_orig = analyzer_orig.analyze_sample(test_data, "Original")
        results_enh = analyzer_enh.analyze_sample(test_data, "Enhanced")
        
        return results_orig, results_enh

class FloodDetectionMetrics:
    """Métricas específicas para detecção de enchentes"""
    
    @staticmethod
    def calculate_flood_probability(vh_vv_ratio, change_intensity, coherence):
        """Calcula probabilidade de enchente baseada em características SAR"""
        
        # Normalizar características
        vh_vv_norm = (vh_vv_ratio - np.mean(vh_vv_ratio)) / np.std(vh_vv_ratio)
        change_norm = (change_intensity - np.mean(change_intensity)) / np.std(change_intensity)
        coherence_norm = (coherence - np.mean(coherence)) / np.std(coherence)
        
        # Regras empíricas para enchente
        # Enchentes: baixa razão VH/VV, alta mudança, baixa coerência
        flood_score = (-vh_vv_norm + change_norm - coherence_norm) / 3
        
        # Converter para probabilidade usando sigmoid
        probability = 1 / (1 + np.exp(-flood_score))
        
        return probability
    
    @staticmethod
    def evaluate_flood_detection(predictions, ground_truth=None):
        """Avalia performance de detecção de enchentes"""
        
        if ground_truth is not None:
            # Se temos ground truth, calcular métricas
            cm = confusion_matrix(ground_truth.flatten(), predictions.flatten())
            print("Confusion Matrix:")
            print(cm)
            
            report = classification_report(
                ground_truth.flatten(), 
                predictions.flatten(),
                target_names=['No Flood', 'Flood']
            )
            print("Classification Report:")
            print(report)
        
        else:
            # Estatísticas descritivas
            flood_pixels = np.sum(predictions)
            total_pixels = predictions.size
            flood_percentage = flood_pixels / total_pixels * 100
            
            print(f"Pixels detectados como enchente: {flood_pixels}")
            print(f"Porcentagem de área alagada: {flood_percentage:.2f}%")
            
            return flood_percentage

if __name__ == "__main__":
    print("Módulo de análise de mudanças SAR carregado.")
    print("Use as classes ChangeDetectionAnalyzer e FloodDetectionMetrics")
    print("para análise detalhada de dados SAR temporais.")
