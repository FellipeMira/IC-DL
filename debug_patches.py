#!/usr/bin/env python3
"""
Script de debug e limpeza para analisar e remover patches irregulares ou problemáticos.
Identifica patches com problemas que podem atrapalhar o treinamento de modelos.
"""

import numpy as np
import rasterio
import os
import shutil
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
from patch_data_improved import PatchConfig, DataNormalizer, PatchQualityMetrics

def debug_patch_validation():
    """Debug da validação de patches"""
    
    # Configuração atual
    config = PatchConfig()
    print(f"Configuração atual:")
    print(f"  - Normalização: {config.normalization}")
    print(f"  - Quality threshold: {config.quality_threshold}")
    print(f"  - Invalid pixel threshold: {config.invalid_pixel_threshold}")
    print(f"  - Patch size: {config.patch_size}")
    print(f"  - Overlap: {config.overlap}")
    
    # Carregar imagem
    image_path = 'data/raw_images/Sentinel_1_ROI_32.tif'
    print(f"\nCarregando imagem: {image_path}")
    
    with rasterio.open(image_path) as src:
        image_data = src.read()  # Shape: [C, H, W]
        image_data = np.transpose(image_data, (1, 2, 0))  # Para [H, W, C]
    
    print(f"Dados originais:")
    print(f"  - Shape: {image_data.shape}")
    print(f"  - Min: {np.min(image_data):.4f}")
    print(f"  - Max: {np.max(image_data):.4f}")
    print(f"  - Mean: {np.mean(image_data):.4f}")
    print(f"  - Std: {np.std(image_data):.4f}")
    print(f"  - Zeros: {np.sum(image_data == 0)}/{image_data.size} ({np.sum(image_data == 0)/image_data.size*100:.2f}%)")
    
    # Aplicar normalização
    normalized_data = DataNormalizer.normalize(image_data, config.normalization)
    
    print(f"\nDados após normalização '{config.normalization}':")
    print(f"  - Min: {np.min(normalized_data):.4f}")
    print(f"  - Max: {np.max(normalized_data):.4f}")
    print(f"  - Mean: {np.mean(normalized_data):.4f}")
    print(f"  - Std: {np.std(normalized_data):.4f}")
    print(f"  - Zeros: {np.sum(normalized_data == 0)}/{normalized_data.size} ({np.sum(normalized_data == 0)/normalized_data.size*100:.2f}%)")
    
    # Testar alguns patches
    height, width = normalized_data.shape[:2]
    stride = int(config.patch_size * (1 - config.overlap))
    
    print(f"\nTestando patches:")
    print(f"  - Stride: {stride}")
    
    patch_count = 0
    valid_patches = 0
    reasons = {}
    quality_scores = []
    
    for i in range(0, height - config.patch_size + 1, stride):
        for j in range(0, width - config.patch_size + 1, stride):
            patch = normalized_data[i:i+config.patch_size, j:j+config.patch_size, :]
            patch_count += 1
            
            # Verificar NaN
            if config.check_nan and np.any(np.isnan(patch)):
                reason = "NaN detectado"
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            
            # Verificar zeros
            total_pixels = patch.size
            zero_pixels = np.sum(patch == 0)
            zero_ratio = zero_pixels / total_pixels
            
            if zero_ratio > config.invalid_pixel_threshold:
                reason = f"Muitos zeros ({zero_ratio:.2%})"
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            
            # Calcular qualidade
            quality_score = PatchQualityMetrics.calculate_quality_score(patch)
            quality_scores.append(quality_score)
            
            if quality_score < config.quality_threshold:
                reason = f"Baixa qualidade ({quality_score:.4f})"
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            
            valid_patches += 1
            
            if patch_count <= 3:  # Detalhar primeiros patches
                print(f"\n  Patch {patch_count} ({i}, {j}):")
                print(f"    - Shape: {patch.shape}")
                print(f"    - Min: {np.min(patch):.4f}")
                print(f"    - Max: {np.max(patch):.4f}")
                print(f"    - Mean: {np.mean(patch):.4f}")
                print(f"    - Zeros: {zero_ratio:.2%}")
                print(f"    - Quality: {quality_score:.4f}")
                print(f"    - Status: {'VÁLIDO' if quality_score >= config.quality_threshold else 'REJEITADO'}")
    
    print(f"\nResumo:")
    print(f"  - Total de patches testados: {patch_count}")
    print(f"  - Patches válidos: {valid_patches}")
    print(f"  - Taxa de aprovação: {valid_patches/patch_count*100:.1f}%")
    
    if quality_scores:
        print(f"  - Quality scores:")
        print(f"    - Min: {np.min(quality_scores):.4f}")
        print(f"    - Max: {np.max(quality_scores):.4f}")
        print(f"    - Mean: {np.mean(quality_scores):.4f}")
        print(f"    - Threshold: {config.quality_threshold}")
    
    print(f"\nRazões de descarte:")
    for reason, count in reasons.items():
        print(f"  - {reason}: {count} patches")
    
    # Sugestões de ajuste
    print(f"\nSugestões de ajuste:")
    
    if quality_scores:
        suggested_threshold = np.percentile(quality_scores, 20)  # 20% melhores
        print(f"  - Reduzir quality_threshold para {suggested_threshold:.4f} (20° percentil)")
    
    print(f"  - Considerar mudança de normalização:")
    print(f"    - 'none': sem normalização (dados brutos)")
    print(f"    - 'log1p': log(1+x) para SAR")
    
    # Testar diferentes normalizações
    print(f"\nTestando diferentes normalizações:")
    for norm_method in ['none', 'log1p', 'zscore']:
        try:
            test_data = DataNormalizer.normalize(image_data, norm_method)
            test_patch = test_data[0:config.patch_size, 0:config.patch_size, :]
            quality = PatchQualityMetrics.calculate_quality_score(test_patch)
            zeros = np.sum(test_patch == 0) / test_patch.size
            
            print(f"  - {norm_method}: quality={quality:.4f}, zeros={zeros:.2%}")
        except Exception as e:
            print(f"  - {norm_method}: ERRO - {str(e)}")


class PatchCleaner:
    """Classe para identificar e remover patches problemáticos"""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.problem_patches = []
        self.backup_dir = os.path.join(data_dir, 'backup_removed_patches')
        
    def analyze_all_patches(self, verbose: bool = True) -> Dict:
        """Analisa todos os patches em busca de problemas"""
        print("🔍 Analisando todos os patches para identificar problemas...")
        
        patch_analysis = {
            'total_patches': 0,
            'problematic_patches': [],
            'statistics': {
                'nan_patches': 0,
                'infinite_patches': 0,
                'zero_heavy_patches': 0,
                'low_quality_patches': 0,
                'corrupted_patches': 0,
                'outlier_patches': 0
            },
            'splits': {}
        }
        
        # Analisar cada split
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.data_dir, split, 'images')
            if not os.path.exists(split_dir):
                continue
                
            patch_files = [f for f in os.listdir(split_dir) if f.endswith('.npy')]
            patch_analysis['splits'][split] = {
                'total': len(patch_files),
                'problems': []
            }
            
            print(f"\n📂 Analisando split '{split}': {len(patch_files)} patches")
            
            for patch_file in patch_files:
                patch_path = os.path.join(split_dir, patch_file)
                problems = self._analyze_single_patch(patch_path, verbose=verbose)
                
                if problems:
                    patch_analysis['problematic_patches'].append({
                        'path': patch_path,
                        'split': split,
                        'filename': patch_file,
                        'problems': problems
                    })
                    patch_analysis['splits'][split]['problems'].extend(problems)
                    
                    # Contar tipos de problemas
                    for problem in problems:
                        if 'NaN' in problem:
                            patch_analysis['statistics']['nan_patches'] += 1
                        elif 'infinito' in problem:
                            patch_analysis['statistics']['infinite_patches'] += 1
                        elif 'zeros' in problem:
                            patch_analysis['statistics']['zero_heavy_patches'] += 1
                        elif 'qualidade' in problem:
                            patch_analysis['statistics']['low_quality_patches'] += 1
                        elif 'corrompido' in problem:
                            patch_analysis['statistics']['corrupted_patches'] += 1
                        elif 'outlier' in problem:
                            patch_analysis['statistics']['outlier_patches'] += 1
                
                patch_analysis['total_patches'] += 1
        
        return patch_analysis
    
    def _analyze_single_patch(self, patch_path: str, verbose: bool = False) -> List[str]:
        """Analisa um único patch em busca de problemas"""
        problems = []
        
        try:
            # Carregar patch
            patch = np.load(patch_path)
            
            # Verificar se o arquivo foi carregado corretamente
            if patch is None or patch.size == 0:
                problems.append("Patch vazio ou corrompido")
                return problems
            
            # Verificar forma esperada
            if len(patch.shape) != 3:
                problems.append(f"Forma inesperada: {patch.shape} (esperado: (H, W, C))")
            
            # Verificar NaN
            if np.any(np.isnan(patch)):
                nan_ratio = np.sum(np.isnan(patch)) / patch.size
                problems.append(f"Contém NaN ({nan_ratio:.2%} dos pixels)")
            
            # Verificar valores infinitos
            if np.any(np.isinf(patch)):
                inf_ratio = np.sum(np.isinf(patch)) / patch.size
                problems.append(f"Contém valores infinitos ({inf_ratio:.2%} dos pixels)")
            
            # Verificar zeros excessivos
            zero_ratio = np.sum(patch == 0) / patch.size
            if zero_ratio > 0.95:  # Mais de 95% zeros
                problems.append(f"Muitos zeros ({zero_ratio:.2%} dos pixels)")
            
            # Verificar se todos os valores são idênticos
            if np.all(patch == patch.flat[0]):
                problems.append("Todos os pixels têm valor idêntico")
            
            # Verificar outliers extremos
            if not np.any(np.isnan(patch)) and not np.any(np.isinf(patch)):
                mean_val = np.mean(patch)
                std_val = np.std(patch)
                
                if std_val > 0:
                    # Verificar se há valores muito extremos (> 5 desvios padrão)
                    outliers = np.abs(patch - mean_val) > (5 * std_val)
                    outlier_ratio = np.sum(outliers) / patch.size
                    
                    if outlier_ratio > 0.1:  # Mais de 10% outliers
                        problems.append(f"Muitos outliers extremos ({outlier_ratio:.2%} dos pixels)")
            
            # Verificar qualidade geral usando as métricas existentes
            try:
                quality_score = PatchQualityMetrics.calculate_quality_score(patch)
                if quality_score < 0.01:  # Qualidade muito baixa
                    problems.append(f"Qualidade muito baixa ({quality_score:.4f})")
            except Exception:
                problems.append("Erro ao calcular qualidade")
            
            # Verificar se valores estão em uma faixa razoável
            if not np.any(np.isnan(patch)) and not np.any(np.isinf(patch)):
                min_val, max_val = np.min(patch), np.max(patch)
                
                # Para dados SAR, valores muito extremos podem indicar problemas
                if max_val - min_val > 1000:  # Range muito grande
                    problems.append(f"Range de valores muito grande: {min_val:.2f} a {max_val:.2f}")
                
                # Verificar se todos os valores são negativos ou positivos demais
                if np.all(patch < -100):
                    problems.append("Todos os valores são muito negativos (< -100)")
                elif np.all(patch > 1000):
                    problems.append("Todos os valores são muito grandes (> 1000)")
            
            if verbose and problems:
                print(f"  ⚠️ {os.path.basename(patch_path)}: {len(problems)} problema(s) encontrado(s)")
                for problem in problems:
                    print(f"    - {problem}")
                    
        except Exception as e:
            problems.append(f"Erro ao carregar patch: {str(e)}")
            if verbose:
                print(f"  ❌ {os.path.basename(patch_path)}: Erro - {str(e)}")
        
        return problems
    
    def create_backup(self, patches_to_remove: List[Dict]) -> str:
        """Cria backup dos patches que serão removidos"""
        if not patches_to_remove:
            return None
            
        # Criar diretório de backup
        os.makedirs(self.backup_dir, exist_ok=True)
        
        backup_info = {
            'backup_timestamp': str(np.datetime64('now')),
            'total_patches_backed_up': len(patches_to_remove),
            'patches': []
        }
        
        print(f"💾 Criando backup de {len(patches_to_remove)} patches em {self.backup_dir}")
        
        for patch_info in patches_to_remove:
            original_path = patch_info['path']
            filename = patch_info['filename']
            split = patch_info['split']
            
            # Criar estrutura de diretórios no backup
            backup_split_dir = os.path.join(self.backup_dir, split, 'images')
            os.makedirs(backup_split_dir, exist_ok=True)
            
            # Copiar arquivo
            backup_path = os.path.join(backup_split_dir, filename)
            shutil.copy2(original_path, backup_path)
            
            backup_info['patches'].append({
                'original_path': original_path,
                'backup_path': backup_path,
                'problems': patch_info['problems']
            })
        
        # Salvar informações do backup
        import yaml
        backup_info_file = os.path.join(self.backup_dir, 'backup_info.yaml')
        with open(backup_info_file, 'w') as f:
            yaml.dump(backup_info, f, default_flow_style=False)
        
        print(f"✅ Backup criado com informações salvas em {backup_info_file}")
        return self.backup_dir
    
    def remove_problematic_patches(self, patch_analysis: Dict, 
                                 auto_confirm: bool = False,
                                 backup: bool = True) -> Dict:
        """Remove patches problemáticos"""
        problematic = patch_analysis['problematic_patches']
        
        if not problematic:
            print("✅ Nenhum patch problemático encontrado!")
            return {'removed': 0, 'backup_dir': None}
        
        print(f"\n🗑️ Encontrados {len(problematic)} patches problemáticos:")
        
        # Mostrar resumo dos problemas
        problem_summary = {}
        for patch_info in problematic:
            for problem in patch_info['problems']:
                problem_summary[problem] = problem_summary.get(problem, 0) + 1
        
        print("\n📊 Resumo dos problemas:")
        for problem, count in sorted(problem_summary.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {problem}: {count} patches")
        
        # Confirmação do usuário
        if not auto_confirm:
            print(f"\n❓ Deseja remover estes {len(problematic)} patches problemáticos?")
            print("   Isso pode melhorar significativamente a qualidade dos dados de treinamento.")
            if backup:
                print(f"   (Um backup será criado em {self.backup_dir})")
            
            response = input("   Confirmar remoção? [s/N]: ").lower().strip()
            if response not in ['s', 'sim', 'y', 'yes']:
                print("❌ Operação cancelada pelo usuário.")
                return {'removed': 0, 'backup_dir': None}
        
        # Criar backup se solicitado
        backup_dir = None
        if backup:
            backup_dir = self.create_backup(problematic)
        
        # Remover patches
        removed_count = 0
        for patch_info in problematic:
            try:
                os.remove(patch_info['path'])
                removed_count += 1
                print(f"  🗑️ Removido: {patch_info['filename']} ({patch_info['split']})")
            except Exception as e:
                print(f"  ❌ Erro ao remover {patch_info['filename']}: {str(e)}")
        
        print(f"\n✅ Remoção concluída: {removed_count} patches removidos")
        
        return {
            'removed': removed_count,
            'backup_dir': backup_dir,
            'problems_found': problem_summary
        }
    
    def _analyze_quality_distribution(self, quality_threshold: float) -> Dict:
        """Analisa distribuição de qualidade dos patches"""
        quality_scores = []
        low_quality_count = 0
        
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.data_dir, split, 'images')
            if not os.path.exists(split_dir):
                continue
                
            patch_files = [f for f in os.listdir(split_dir) if f.endswith('.npy')]
            
            for patch_file in patch_files:
                patch_path = os.path.join(split_dir, patch_file)
                
                try:
                    patch = np.load(patch_path)
                    quality_score = PatchQualityMetrics.calculate_quality_score(patch)
                    quality_scores.append(quality_score)
                    
                    if quality_score < quality_threshold:
                        low_quality_count += 1
                        
                except Exception:
                    continue
        
        return {
            'low_quality_count': low_quality_count,
            'total_patches': len(quality_scores),
            'mean_quality': np.mean(quality_scores) if quality_scores else 0,
            'median_quality': np.median(quality_scores) if quality_scores else 0,
            'min_quality': np.min(quality_scores) if quality_scores else 0,
            'max_quality': np.max(quality_scores) if quality_scores else 0,
            'quality_scores': quality_scores
        }
    
    def optimize_dataset_quality(self, quality_threshold: float = 1.0,
                                auto_confirm: bool = False) -> Dict:
        """Remove patches de baixa qualidade para otimizar o dataset"""
        print(f"🎯 Otimizando qualidade do dataset (threshold: {quality_threshold})")
        
        low_quality_patches = []
        quality_scores = []
        
        # Analisar qualidade de todos os patches
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.data_dir, split, 'images')
            if not os.path.exists(split_dir):
                continue
                
            patch_files = [f for f in os.listdir(split_dir) if f.endswith('.npy')]
            
            for patch_file in patch_files:
                patch_path = os.path.join(split_dir, patch_file)
                
                try:
                    patch = np.load(patch_path)
                    quality_score = PatchQualityMetrics.calculate_quality_score(patch)
                    quality_scores.append(quality_score)
                    
                    if quality_score < quality_threshold:
                        low_quality_patches.append({
                            'path': patch_path,
                            'split': split,
                            'filename': patch_file,
                            'quality_score': quality_score,
                            'problems': [f"Baixa qualidade ({quality_score:.4f})"]
                        })
                        
                except Exception as e:
                    print(f"  ❌ Erro ao analisar {patch_file}: {str(e)}")
        
        if quality_scores:
            print(f"📈 Estatísticas de qualidade:")
            print(f"  - Média: {np.mean(quality_scores):.4f}")
            print(f"  - Mediana: {np.median(quality_scores):.4f}")
            print(f"  - Min: {np.min(quality_scores):.4f}")
            print(f"  - Max: {np.max(quality_scores):.4f}")
            print(f"  - Patches abaixo do threshold ({quality_threshold}): {len(low_quality_patches)}")
        
        if not low_quality_patches:
            print("✅ Todos os patches atendem ao critério de qualidade!")
            return {'removed': 0, 'backup_dir': None}
        
        # Criar análise fake para usar o método de remoção existente
        fake_analysis = {'problematic_patches': low_quality_patches}
        
        return self.remove_problematic_patches(fake_analysis, auto_confirm=auto_confirm)


def main():
    """Função principal com interface de linha de comando"""
    parser = argparse.ArgumentParser(
        description='Debug e limpeza de patches SAR problemáticos'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='data',
        help='Diretório com os dados (default: data)'
    )
    parser.add_argument(
        '--analyze-only', '-a',
        action='store_true',
        help='Apenas analisar, não remover patches'
    )
    parser.add_argument(
        '--auto-confirm', '-y',
        action='store_true',
        help='Confirmar automaticamente a remoção'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Não criar backup dos patches removidos'
    )
    parser.add_argument(
        '--quality-threshold', '-q',
        type=float,
        help='Threshold mínimo de qualidade para otimização'
    )
    parser.add_argument(
        '--debug-creation',
        action='store_true',
        help='Executar debug de criação de patches'
    )
    
    args = parser.parse_args()
    
    if args.debug_creation:
        # Executar debug original de criação
        debug_patch_validation()
        return 0
    
    try:
        # Verificar se diretório existe
        if not os.path.exists(args.data_dir):
            print(f"❌ Diretório '{args.data_dir}' não encontrado!")
            return 1
        
        # Criar limpador de patches
        cleaner = PatchCleaner(args.data_dir)
        
        # Analisar patches
        print("🔍 ANÁLISE DE PATCHES PROBLEMÁTICOS")
        print("="*50)
        analysis = cleaner.analyze_all_patches(verbose=True)
        
        # Mostrar resultados da análise
        print(f"\n📊 RESUMO DA ANÁLISE:")
        print(f"  Total de patches analisados: {analysis['total_patches']}")
        print(f"  Patches problemáticos: {len(analysis['problematic_patches'])}")
        print(f"  Taxa de problemas: {len(analysis['problematic_patches'])/analysis['total_patches']*100:.1f}%")
        
        if analysis['statistics']:
            print(f"\n🏷️ TIPOS DE PROBLEMAS:")
            stats = analysis['statistics']
            for problem_type, count in stats.items():
                if count > 0:
                    print(f"  - {problem_type.replace('_', ' ').title()}: {count}")
        
        print(f"\n📂 POR SPLIT:")
        for split, split_info in analysis['splits'].items():
            problem_count = len(set(split_info['problems']))
            print(f"  - {split}: {split_info['total']} patches, {problem_count} com problemas")
        
        # Se apenas análise, parar aqui
        if args.analyze_only:
            if args.quality_threshold is not None:
                print(f"\n🎯 ANÁLISE DE QUALIDADE (threshold: {args.quality_threshold})")
                print("="*50)
                
                # Analisar qualidade sem remover
                quality_analysis = cleaner._analyze_quality_distribution(args.quality_threshold)
                print(f"  Patches abaixo do threshold: {quality_analysis['low_quality_count']}")
                print(f"  Qualidade média: {quality_analysis['mean_quality']:.4f}")
                print(f"  Qualidade mediana: {quality_analysis['median_quality']:.4f}")
                print(f"  Range: {quality_analysis['min_quality']:.4f} - {quality_analysis['max_quality']:.4f}")
            
            print("\n✅ Análise concluída. Remova --analyze-only para executar limpeza.")
            return 0
        
        # Remover patches problemáticos
        if len(analysis['problematic_patches']) > 0:
            print(f"\n🗑️ REMOÇÃO DE PATCHES PROBLEMÁTICOS")
            print("="*50)
            
            removal_result = cleaner.remove_problematic_patches(
                analysis, 
                auto_confirm=args.auto_confirm,
                backup=not args.no_backup
            )
            
            print(f"\n✅ RESULTADO:")
            print(f"  Patches removidos: {removal_result['removed']}")
            if removal_result['backup_dir']:
                print(f"  Backup criado em: {removal_result['backup_dir']}")
        
        # Otimização de qualidade se threshold especificado
        if args.quality_threshold is not None:
            print(f"\n🎯 OTIMIZAÇÃO DE QUALIDADE")
            print("="*50)
            
            optimization_result = cleaner.optimize_dataset_quality(
                quality_threshold=args.quality_threshold,
                auto_confirm=args.auto_confirm
            )
            
            print(f"\n✅ OTIMIZAÇÃO CONCLUÍDA:")
            print(f"  Patches de baixa qualidade removidos: {optimization_result['removed']}")
        
        print(f"\n🎉 LIMPEZA CONCLUÍDA!")
        print("Seu dataset agora possui apenas patches de alta qualidade para treinamento.")
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
