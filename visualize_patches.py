"""
Script para visualização e análise dos patches criados.
Permite verificar a qualidade e distribuição dos dados gerados.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from typing import List, Dict, Optional
import argparse
from datetime import datetime


class PatchVisualizer:
    """Classe para visualização e análise de patches SAR"""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.patches_info = self._load_patches_info()
    
    def _load_patches_info(self) -> Dict:
        """Carrega informações sobre os patches disponíveis"""
        patches_info = {
            'train': [],
            'val': [],
            'test': []
        }
        
        for split in patches_info.keys():
            split_dir = os.path.join(self.data_dir, split, 'images')
            if os.path.exists(split_dir):
                patches = [f for f in os.listdir(split_dir) if f.endswith('.npy')]
                patches_info[split] = [os.path.join(split_dir, p) for p in patches]
        
        return patches_info
    
    def load_patch(self, patch_path: str) -> np.ndarray:
        """Carrega um patch do disco"""
        try:
            patch = np.load(patch_path)
            return patch
        except Exception as e:
            print(f"Erro ao carregar patch {patch_path}: {str(e)}")
            return None
    
    def visualize_patch(self, patch: np.ndarray, title: str = "", 
                       channels_to_show: List[int] = None, 
                       figsize: tuple = (12, 8)):
        """Visualiza um patch SAR"""
        if patch is None:
            print("Patch inválido")
            return None
        
        # Selecionar canais para visualização
        if channels_to_show is None:
            # Mostrar primeiros 4 canais ou todos se menos de 4
            n_channels = min(4, patch.shape[2])
            channels_to_show = list(range(n_channels))
        
        n_channels = len(channels_to_show)
        
        # Configurar subplot
        cols = min(4, n_channels)
        rows = (n_channels + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle(title, fontsize=14)
        
        if n_channels == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if n_channels == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, channel in enumerate(channels_to_show):
            if i >= len(axes):
                break
                
            ax = axes[i] if n_channels > 1 else axes[0]
            
            # Extrair canal
            channel_data = patch[:, :, channel]
            
            # Visualizar
            im = ax.imshow(channel_data, cmap='gray', interpolation='nearest')
            ax.set_title(f'Canal {channel}')
            ax.axis('off')
            
            # Estatísticas
            stats_text = (f'Min: {np.min(channel_data):.3f}\n'
                         f'Max: {np.max(channel_data):.3f}\n'
                         f'Mean: {np.mean(channel_data):.3f}\n'
                         f'Std: {np.std(channel_data):.3f}')
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8), fontsize=8)
            
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        # Remover eixos vazios
        for i in range(n_channels, len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        return fig
    
    def analyze_patch_statistics(self, split: str = 'train') -> Dict:
        """Analisa estatísticas dos patches de um split"""
        patches = self.patches_info[split]
        
        if not patches:
            print(f"Nenhum patch encontrado para split '{split}'")
            return {}
        
        stats = {
            'count': len(patches),
            'shapes': [],
            'means': [],
            'stds': [],
            'mins': [],
            'maxs': [],
            'channels': []
        }
        
        print(f"Analisando {len(patches)} patches do split '{split}'...")
        
        for i, patch_path in enumerate(patches):
            patch = self.load_patch(patch_path)
            if patch is not None:
                stats['shapes'].append(patch.shape)
                stats['means'].append(np.mean(patch))
                stats['stds'].append(np.std(patch))
                stats['mins'].append(np.min(patch))
                stats['maxs'].append(np.max(patch))
                stats['channels'].append(patch.shape[2])
            
            if (i + 1) % 10 == 0:
                print(f"Processados {i + 1}/{len(patches)} patches")
        
        # Calcular estatísticas consolidadas
        if stats['means']:
            stats['global_stats'] = {
                'mean_of_means': np.mean(stats['means']),
                'std_of_means': np.std(stats['means']),
                'min_value': np.min(stats['mins']),
                'max_value': np.max(stats['maxs']),
                'mean_std': np.mean(stats['stds']),
                'common_shape': max(set(map(tuple, stats['shapes'])), 
                                  key=stats['shapes'].count),
                'common_channels': max(set(stats['channels']), 
                                     key=stats['channels'].count)
            }
        
        return stats
    
    def plot_statistics(self, stats: Dict, split: str = 'train', 
                       figsize: tuple = (15, 10)):
        """Plota estatísticas dos patches"""
        if not stats or 'global_stats' not in stats:
            print("Estatísticas inválidas")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Estatísticas dos Patches - Split: {split}', fontsize=16)
        
        # Histograma das médias
        axes[0, 0].hist(stats['means'], bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_title('Distribuição das Médias')
        axes[0, 0].set_xlabel('Média')
        axes[0, 0].set_ylabel('Frequência')
        
        # Histograma dos desvios padrão
        axes[0, 1].hist(stats['stds'], bins=30, alpha=0.7, color='green')
        axes[0, 1].set_title('Distribuição dos Desvios Padrão')
        axes[0, 1].set_xlabel('Desvio Padrão')
        axes[0, 1].set_ylabel('Frequência')
        
        # Distribuição de valores mínimos e máximos
        axes[0, 2].hist(stats['mins'], bins=30, alpha=0.5, color='red', label='Mínimos')
        axes[0, 2].hist(stats['maxs'], bins=30, alpha=0.5, color='orange', label='Máximos')
        axes[0, 2].set_title('Distribuição Min/Max')
        axes[0, 2].set_xlabel('Valor')
        axes[0, 2].set_ylabel('Frequência')
        axes[0, 2].legend()
        
        # Scatter plot: média vs desvio padrão
        axes[1, 0].scatter(stats['means'], stats['stds'], alpha=0.6)
        axes[1, 0].set_title('Média vs Desvio Padrão')
        axes[1, 0].set_xlabel('Média')
        axes[1, 0].set_ylabel('Desvio Padrão')
        
        # Distribuição de formas
        shapes_count = {}
        for shape in stats['shapes']:
            shape_str = f"{shape[0]}x{shape[1]}x{shape[2]}"
            shapes_count[shape_str] = shapes_count.get(shape_str, 0) + 1
        
        shapes_labels = list(shapes_count.keys())
        shapes_values = list(shapes_count.values())
        
        axes[1, 1].bar(range(len(shapes_labels)), shapes_values)
        axes[1, 1].set_title('Distribuição de Formas')
        axes[1, 1].set_xlabel('Forma (HxWxC)')
        axes[1, 1].set_ylabel('Quantidade')
        axes[1, 1].set_xticks(range(len(shapes_labels)))
        axes[1, 1].set_xticklabels(shapes_labels, rotation=45)
        
        # Texto com estatísticas globais
        global_stats = stats['global_stats']
        stats_text = (
            f"Total de patches: {stats['count']}\n"
            f"Forma comum: {global_stats['common_shape']}\n"
            f"Canais comuns: {global_stats['common_channels']}\n"
            f"Média global: {global_stats['mean_of_means']:.4f}\n"
            f"Desvio das médias: {global_stats['std_of_means']:.4f}\n"
            f"Valor mínimo: {global_stats['min_value']:.4f}\n"
            f"Valor máximo: {global_stats['max_value']:.4f}\n"
            f"Desvio médio: {global_stats['mean_std']:.4f}"
        )
        
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 2].set_title('Estatísticas Globais')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_random_samples(self, split: str = 'train', n_samples: int = 6,
                               channels_per_patch: int = 2, figsize: tuple = (18, 12)):
        """Visualiza amostras aleatórias de patches"""
        patches = self.patches_info[split]
        
        if not patches:
            print(f"Nenhum patch encontrado para split '{split}'")
            return None
        
        # Selecionar amostras aleatórias
        np.random.seed(42)
        sample_indices = np.random.choice(len(patches), 
                                        size=min(n_samples, len(patches)), 
                                        replace=False)
        
        fig, axes = plt.subplots(n_samples, channels_per_patch, figsize=figsize)
        fig.suptitle(f'Amostras Aleatórias - Split: {split}', fontsize=16)
        
        if n_samples == 1:
            axes = [axes]
        
        for i, idx in enumerate(sample_indices):
            patch_path = patches[idx]
            patch = self.load_patch(patch_path)
            
            if patch is None:
                continue
            
            patch_name = os.path.basename(patch_path)
            
            # Selecionar canais para mostrar
            available_channels = patch.shape[2]
            channels_to_show = np.linspace(0, available_channels-1, 
                                         channels_per_patch, dtype=int)
            
            for j, channel in enumerate(channels_to_show):
                if channels_per_patch == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j] if n_samples > 1 else axes[j]
                
                channel_data = patch[:, :, channel]
                
                im = ax.imshow(channel_data, cmap='viridis', interpolation='nearest')
                ax.set_title(f'{patch_name}\nCanal {channel}', fontsize=10)
                ax.axis('off')
                
                # Adicionar colorbar
                plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        return fig
    
    def compare_splits(self, figsize: tuple = (15, 8)):
        """Compara estatísticas entre splits"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Comparação entre Splits', fontsize=16)
        
        splits_stats = {}
        colors = ['blue', 'green', 'red']
        
        # Calcular estatísticas para cada split
        for split in ['train', 'val', 'test']:
            if self.patches_info[split]:
                print(f"Calculando estatísticas para {split}...")
                splits_stats[split] = self.analyze_patch_statistics(split)
        
        if not splits_stats:
            print("Nenhuma estatística disponível")
            return None
        
        # Plot 1: Distribuição das médias
        for i, (split, stats) in enumerate(splits_stats.items()):
            if 'means' in stats and stats['means']:
                axes[0, 0].hist(stats['means'], bins=20, alpha=0.6, 
                              color=colors[i], label=f'{split} (n={stats["count"]})')
        
        axes[0, 0].set_title('Distribuição das Médias por Split')
        axes[0, 0].set_xlabel('Média')
        axes[0, 0].set_ylabel('Frequência')
        axes[0, 0].legend()
        
        # Plot 2: Boxplot das médias
        means_data = []
        labels = []
        for split, stats in splits_stats.items():
            if 'means' in stats and stats['means']:
                means_data.append(stats['means'])
                labels.append(f'{split} (n={stats["count"]})')
        
        if means_data:
            axes[0, 1].boxplot(means_data, labels=labels)
            axes[0, 1].set_title('Boxplot das Médias')
            axes[0, 1].set_ylabel('Média')
        
        # Plot 3: Comparação de estatísticas globais
        global_means = []
        global_stds = []
        split_names = []
        
        for split, stats in splits_stats.items():
            if 'global_stats' in stats:
                global_means.append(stats['global_stats']['mean_of_means'])
                global_stds.append(stats['global_stats']['mean_std'])
                split_names.append(split)
        
        if global_means:
            x_pos = range(len(split_names))
            axes[1, 0].bar([x - 0.2 for x in x_pos], global_means, 
                          width=0.4, label='Média Global', alpha=0.7)
            axes[1, 0].bar([x + 0.2 for x in x_pos], global_stds, 
                          width=0.4, label='Desvio Médio', alpha=0.7)
            axes[1, 0].set_title('Estatísticas Globais por Split')
            axes[1, 0].set_xlabel('Split')
            axes[1, 0].set_ylabel('Valor')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(split_names)
            axes[1, 0].legend()
        
        # Plot 4: Contagem de patches
        patch_counts = [len(self.patches_info[split]) for split in ['train', 'val', 'test']]
        split_labels = ['Train', 'Validation', 'Test']
        
        axes[1, 1].pie(patch_counts, labels=split_labels, autopct='%1.1f%%', 
                      colors=colors[:3])
        axes[1, 1].set_title('Distribuição de Patches por Split')
        
        plt.tight_layout()
        return fig
    
    def save_analysis_report(self, output_dir: str = 'analysis_reports'):
        """Gera relatório completo de análise"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("Gerando relatório de análise...")
        
        # Análise por split
        all_stats = {}
        for split in ['train', 'val', 'test']:
            if self.patches_info[split]:
                print(f"Analisando split: {split}")
                stats = self.analyze_patch_statistics(split)
                all_stats[split] = stats
                
                # Plotar estatísticas
                if stats:
                    fig = self.plot_statistics(stats, split)
                    if fig:
                        fig.savefig(os.path.join(output_dir, 
                                               f'statistics_{split}_{timestamp}.png'),
                                  dpi=300, bbox_inches='tight')
                        plt.close(fig)
                
                # Visualizar amostras
                fig = self.visualize_random_samples(split, n_samples=4)
                if fig:
                    fig.savefig(os.path.join(output_dir, 
                                           f'samples_{split}_{timestamp}.png'),
                              dpi=300, bbox_inches='tight')
                    plt.close(fig)
        
        # Comparação entre splits
        if len(all_stats) > 1:
            fig = self.compare_splits()
            if fig:
                fig.savefig(os.path.join(output_dir, 
                                       f'comparison_{timestamp}.png'),
                          dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        # Salvar estatísticas em YAML
        report_data = {
            'analysis_timestamp': timestamp,
            'data_directory': self.data_dir,
            'statistics': all_stats,
            'summary': {
                'total_patches': sum(len(patches) for patches in self.patches_info.values()),
                'splits_available': list(all_stats.keys()),
                'splits_distribution': {split: len(patches) 
                                      for split, patches in self.patches_info.items()}
            }
        }
        
        report_file = os.path.join(output_dir, f'analysis_report_{timestamp}.yaml')
        with open(report_file, 'w') as f:
            yaml.dump(report_data, f, default_flow_style=False)
        
        print(f"Relatório salvo em: {output_dir}")
        print(f"Arquivo principal: {report_file}")
        
        return report_file


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='Visualização e análise de patches SAR'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='data',
        help='Diretório com os dados (default: data)'
    )
    parser.add_argument(
        '--split', '-s',
        type=str,
        choices=['train', 'val', 'test'],
        help='Split específico para análise'
    )
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=6,
        help='Número de amostras para visualização (default: 6)'
    )
    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='Gerar relatório completo'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='analysis_reports',
        help='Diretório para salvar relatórios (default: analysis_reports)'
    )
    
    args = parser.parse_args()
    
    try:
        # Criar visualizador
        visualizer = PatchVisualizer(args.data_dir)
        
        # Verificar se há dados
        total_patches = sum(len(patches) for patches in visualizer.patches_info.values())
        if total_patches == 0:
            print(f"Nenhum patch encontrado em {args.data_dir}")
            return 1
        
        print(f"Encontrados {total_patches} patches:")
        for split, patches in visualizer.patches_info.items():
            print(f"  {split}: {len(patches)} patches")
        
        if args.report:
            # Gerar relatório completo
            visualizer.save_analysis_report(args.output_dir)
        
        elif args.split:
            # Análise de split específico
            stats = visualizer.analyze_patch_statistics(args.split)
            if stats:
                fig1 = visualizer.plot_statistics(stats, args.split)
                fig2 = visualizer.visualize_random_samples(args.split, args.samples)
                plt.show()
        
        else:
            # Comparação entre splits
            fig = visualizer.compare_splits()
            if fig:
                plt.show()
    
    except Exception as e:
        print(f"Erro: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
