#!/usr/bin/env python3
"""
Script simples e rápido para visualizar logs de treinamento da rede SAR
Suporta múltiplas fontes: logs de texto, checkpoints PyTorch e execução ao vivo
"""

import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from datetime import datetime

def extract_from_checkpoint(checkpoint_path):
    """Extrai histórico de loss do checkpoint PyTorch"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        history = {}
        # Procurar por diferentes chaves possíveis no checkpoint
        possible_keys = [
            'train_losses', 'val_losses', 'train_loss_history', 'val_loss_history',
            'loss_history', 'history', 'metrics', 'epoch_losses'
        ]
        
        for key in possible_keys:
            if key in checkpoint:
                history[key] = checkpoint[key]
        
        # Se não encontrar histórico, pelo menos mostrar info do modelo
        if not history:
            print(f"📦 Checkpoint: {os.path.basename(checkpoint_path)}")
            if 'epoch' in checkpoint:
                print(f"   • Época: {checkpoint['epoch']}")
            if 'best_val_loss' in checkpoint:
                print(f"   • Melhor Val Loss: {checkpoint['best_val_loss']:.6f}")
            if 'train_loss' in checkpoint:
                print(f"   • Train Loss: {checkpoint['train_loss']:.6f}")
            if 'val_loss' in checkpoint:
                print(f"   • Val Loss: {checkpoint['val_loss']:.6f}")
            return None
            
        return history
        
    except Exception as e:
        print(f"❌ Erro ao carregar checkpoint {checkpoint_path}: {e}")
        return None

def extract_from_logs(log_pattern="training_log_*.log"):
    """Extrai dados de treinamento dos arquivos de log"""
    log_files = glob(log_pattern)
    
    if not log_files:
        return None
    
    print(f"📋 Encontrados {len(log_files)} arquivo(s) de log")
    
    # Usar o log mais recente
    latest_log = max(log_files, key=os.path.getctime)
    print(f"   • Usando: {latest_log}")
    
    epochs = []
    train_losses = []
    val_losses = []
    train_recon = []
    train_temporal = []
    val_recon = []
    val_temporal = []
    learning_rates = []
    
    try:
        with open(latest_log, 'r') as f:
            current_epoch = None
            current_lr = None
            
            for line in f:
                # Padrão 1: Epoch X/Y - LR: Z
                lr_match = re.search(r'Epoch (\d+)/\d+ - LR: ([\d.e-]+)', line)
                if lr_match:
                    current_epoch = int(lr_match.group(1))
                    current_lr = float(lr_match.group(2))
                    continue
                
                # Padrão 2: Train - Total: X, Recon: Y, Temporal: Z
                train_match = re.search(r'Train - Total: ([\d.]+), Recon: ([\d.]+), Temporal: ([\d.]+)', line)
                if train_match and current_epoch is not None:
                    train_total = float(train_match.group(1))
                    train_rec = float(train_match.group(2))
                    train_temp = float(train_match.group(3))
                    continue
                
                # Padrão 3: Val - Total: X, Recon: Y, Temporal: Z
                val_match = re.search(r'Val - Total: ([\d.]+), Recon: ([\d.]+), Temporal: ([\d.]+)', line)
                if val_match and current_epoch is not None:
                    val_total = float(val_match.group(1))
                    val_rec = float(val_match.group(2))
                    val_temp = float(val_match.group(3))
                    
                    # Adicionar dados da época completa
                    epochs.append(current_epoch)
                    train_losses.append(train_total)
                    val_losses.append(val_total)
                    train_recon.append(train_rec)
                    train_temporal.append(train_temp)
                    val_recon.append(val_rec)
                    val_temporal.append(val_temp)
                    learning_rates.append(current_lr)
                    
                    # Reset para próxima época
                    current_epoch = None
                    current_lr = None
        
        if epochs:
            print(f"✅ Extraídas {len(epochs)} épocas do log")
            return {
                'epochs': epochs,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_recon': train_recon,
                'train_temporal': train_temporal,
                'val_recon': val_recon,
                'val_temporal': val_temporal,
                'learning_rates': learning_rates
            }
    
    except Exception as e:
        print(f"❌ Erro ao ler log {latest_log}: {e}")
    
    return None

def plot_training_history(data, source_name="Treinamento"):
    """Cria visualização do histórico de treinamento"""
    
    if not data:
        print("❌ Nenhum dado para plotar")
        return
    
    # Configurar figura com 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Histórico de Treinamento SAR Autoencoder - {source_name}', fontsize=16, fontweight='bold')
    
    epochs = data.get('epochs', range(1, len(data['train_losses']) + 1))
    
    # Plot 1: Total Losses
    if 'train_losses' in data and 'val_losses' in data:
        axes[0,0].plot(epochs, data['train_losses'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
        axes[0,0].plot(epochs, data['val_losses'], 'r-', label='Val Loss', linewidth=2, marker='s', markersize=3)
        axes[0,0].set_xlabel('Época')
        axes[0,0].set_ylabel('Loss Total')
        axes[0,0].set_title('Loss Total (Train vs Val)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Estatísticas
        min_train = min(data['train_losses'])
        min_val = min(data['val_losses'])
        final_train = data['train_losses'][-1]
        final_val = data['val_losses'][-1]
        
        # Texto com estatísticas
        stats_text = f'Mín Train: {min_train:.4f}\nMín Val: {min_val:.4f}\nFinal Train: {final_train:.4f}\nFinal Val: {final_val:.4f}'
        axes[0,0].text(0.02, 0.98, stats_text, transform=axes[0,0].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 2: Reconstruction Losses
    if 'train_recon' in data and 'val_recon' in data:
        axes[0,1].plot(epochs, data['train_recon'], 'g-', label='Train Recon', linewidth=2, marker='o', markersize=3)
        axes[0,1].plot(epochs, data['val_recon'], 'orange', label='Val Recon', linewidth=2, marker='s', markersize=3)
        axes[0,1].set_xlabel('Época')
        axes[0,1].set_ylabel('Reconstruction Loss')
        axes[0,1].set_title('Loss de Reconstrução')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    else:
        axes[0,1].text(0.5, 0.5, 'Dados de Reconstrução\nnão disponíveis', 
                    ha='center', va='center', transform=axes[0,1].transAxes, fontsize=12)
    
    # Plot 3: Temporal Losses
    if 'train_temporal' in data and 'val_temporal' in data:
        axes[1,0].plot(epochs, data['train_temporal'], 'purple', label='Train Temporal', linewidth=2, marker='o', markersize=3)
        axes[1,0].plot(epochs, data['val_temporal'], 'brown', label='Val Temporal', linewidth=2, marker='s', markersize=3)
        axes[1,0].set_xlabel('Época')
        axes[1,0].set_ylabel('Temporal Loss')
        axes[1,0].set_title('Loss Temporal')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    else:
        axes[1,0].text(0.5, 0.5, 'Dados Temporais\nnão disponíveis', 
                    ha='center', va='center', transform=axes[1,0].transAxes, fontsize=12)
    
    # Plot 4: Learning Rate e Overfitting Analysis
    if 'learning_rates' in data:
        ax_lr = axes[1,1]
        ax_diff = ax_lr.twinx()
        
        # Learning rate
        line1 = ax_lr.plot(epochs, data['learning_rates'], 'g-', linewidth=2, marker='d', markersize=3, label='Learning Rate')
        ax_lr.set_xlabel('Época')
        ax_lr.set_ylabel('Learning Rate', color='g')
        ax_lr.set_yscale('log')
        ax_lr.tick_params(axis='y', labelcolor='g')
        ax_lr.grid(True, alpha=0.3)
        
        # Diferença de loss (overfitting)
        if 'train_losses' in data and 'val_losses' in data:
            loss_diff = np.array(data['val_losses']) - np.array(data['train_losses'])
            line2 = ax_diff.plot(epochs, loss_diff, 'red', linewidth=2, alpha=0.7, label='Val-Train Diff')
            ax_diff.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax_diff.set_ylabel('Val Loss - Train Loss', color='red')
            ax_diff.tick_params(axis='y', labelcolor='red')
            
            # Combinar legendas
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax_lr.legend(lines, labels, loc='upper right')
        
        axes[1,1].set_title('Learning Rate & Overfitting')
    else:
        axes[1,1].text(0.5, 0.5, 'Learning Rate\nnão disponível', 
                    ha='center', va='center', transform=axes[1,1].transAxes, fontsize=12)
    
    plt.tight_layout()
    
    # Salvar figura
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'training_visualization_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"💾 Visualização salva: {filename}")
    
    plt.show()

def main():
    """Função principal - busca e visualiza dados de treinamento"""
    print("🔍 Procurando dados de treinamento...")
    
    data = None
    source = ""
    
    # 1. Tentar extrair de logs primeiro
    log_data = extract_from_logs()
    if log_data:
        data = log_data
        source = "Log Files"
        print("✅ Dados extraídos dos logs!")
    
    # 2. Se não tiver logs, tentar checkpoints
    if not data:
        checkpoint_files = ['best_sar_model.pth', 'images/best_sar_model.pth']
        for checkpoint_path in checkpoint_files:
            if os.path.exists(checkpoint_path):
                print(f"🔍 Verificando checkpoint: {checkpoint_path}")
                checkpoint_data = extract_from_checkpoint(checkpoint_path)
                if checkpoint_data:
                    data = checkpoint_data
                    source = f"Checkpoint ({os.path.basename(checkpoint_path)})"
                    print("✅ Dados extraídos do checkpoint!")
                    break
    
    # 3. Se ainda não tiver dados, mostrar instruções
    if not data:
        print("\n❌ Nenhum dado de treinamento encontrado!")
        print("\n📝 Para capturar dados de treinamento:")
        print("   1. Certifique-se que o treinamento salve logs ou histórico no checkpoint")
        print("   2. Execute o treinamento com logs habilitados")
        print("   3. Os logs devem estar no formato: 'training_log_*.log'")
        print("   4. Ou salve 'train_losses' e 'val_losses' no checkpoint")
        return
    
    # 4. Plotar dados encontrados
    print(f"📊 Visualizando dados de: {source}")
    plot_training_history(data, source)
    
    # 5. Resumo final
    if 'train_losses' in data and 'val_losses' in data:
        print(f"\n📈 RESUMO DO TREINAMENTO SAR:")
        print(f"   • Épocas: {len(data['train_losses'])}")
        print(f"   • Melhor Train Loss: {min(data['train_losses']):.6f}")
        print(f"   • Melhor Val Loss: {min(data['val_losses']):.6f}")
        print(f"   • Loss final (Train): {data['train_losses'][-1]:.6f}")
        print(f"   • Loss final (Val): {data['val_losses'][-1]:.6f}")
        
        # Análise de componentes se disponível
        if 'train_recon' in data:
            print(f"   • Reconstrução final (Train): {data['train_recon'][-1]:.6f}")
            print(f"   • Reconstrução final (Val): {data['val_recon'][-1]:.6f}")
        if 'train_temporal' in data:
            print(f"   • Temporal final (Train): {data['train_temporal'][-1]:.6f}")
            print(f"   • Temporal final (Val): {data['val_temporal'][-1]:.6f}")
        
        # Análise simples de overfitting
        final_diff = data['val_losses'][-1] - data['train_losses'][-1]
        best_epoch = data['epochs'][np.argmin(data['val_losses'])]
        improvement = data['val_losses'][0] - min(data['val_losses'])
        improvement_pct = (improvement / data['val_losses'][0]) * 100
        
        print(f"\n🔍 ANÁLISE:")
        print(f"   • Melhor época: {best_epoch}")
        print(f"   • Melhoria total: {improvement:.6f} ({improvement_pct:.1f}%)")
        
        if final_diff > 0.05:
            print(f"   ⚠️  Possível overfitting detectado (diff: {final_diff:.4f})")
        elif final_diff > 0.02:
            print(f"   📊 Leve overfitting (diff: {final_diff:.4f}) - aceitável")
        else:
            print(f"   ✅ Modelo bem ajustado (diff: {final_diff:.4f})")
        
        # Tendência das últimas épocas
        last_5_train = data['train_losses'][-5:]
        last_5_val = data['val_losses'][-5:]
        
        if len(last_5_val) >= 3:
            train_trend = "decrescendo" if last_5_train[-1] < last_5_train[0] else "crescendo" 
            val_trend = "decrescendo" if last_5_val[-1] < last_5_val[0] else "crescendo"
            print(f"   📉 Tendência (últimas 5 épocas): Train {train_trend}, Val {val_trend}")
        
        if 'learning_rates' in data:
            print(f"   🎯 Learning rate final: {data['learning_rates'][-1]:.2e}")

if __name__ == '__main__':
    main()
