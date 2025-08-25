"""
Script melhorado para criação de patches de imagens SAR para treinamento de redes neurais.
Autor: GitHub Copilot
Data: 2025-08-25

Melhorias implementadas:
- Correção de bugs no descarte de patches
- Estrutura orientada a objetos
- Configurações flexíveis via YAML
- Paralelização do processamento
- Métricas de qualidade de patches
- Múltiplos tipos de normalização
- Visualização de progresso
- Melhor tratamento de erros
"""

import os
import numpy as np
import rasterio
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import yaml
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import time
from datetime import datetime
import argparse
import shutil


@dataclass
class PatchConfig:
    """Configurações para criação de patches"""
    patch_size: int = 128
    overlap: float = 0.3
    input_dir: str = 'data/raw_images'
    output_dir: str = 'data'
    split_ratios: Dict[str, float] = None
    seed: int = 42
    expected_channels: Optional[int] = None
    default_split: str = 'train'
    invalid_pixel_threshold: float = 0.95  # Mais permissivo
    check_nan: bool = True
    normalization: str = 'log1p'  # 'none', 'log1p', 'zscore', 'minmax'
    quality_threshold: float = 0.1  # Threshold de qualidade do patch
    max_workers: int = 4
    save_metadata: bool = True
    
    def __post_init__(self):
        if self.split_ratios is None:
            self.split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
        self.stride = int(self.patch_size * (1 - self.overlap))


class PatchQualityMetrics:
    """Calcula métricas de qualidade para patches SAR"""
    
    @staticmethod
    def calculate_texture_energy(patch: np.ndarray) -> float:
        """Calcula a energia de textura usando variância local"""
        # Aplicar filtro de variância local em uma janela 3x3
        from scipy.ndimage import generic_filter
        variance_filter = lambda x: np.var(x)
        
        # Usar apenas a primeira banda para cálculo de textura
        if len(patch.shape) == 3:
            patch_2d = patch[:, :, 0]
        else:
            patch_2d = patch
            
        texture_energy = generic_filter(patch_2d, variance_filter, size=3)
        return np.mean(texture_energy)
    
    @staticmethod
    def calculate_speckle_index(patch: np.ndarray) -> float:
        """Calcula índice de speckle (razão entre desvio padrão e média)"""
        if len(patch.shape) == 3:
            patch_2d = patch[:, :, 0]
        else:
            patch_2d = patch
            
        mean_val = np.mean(patch_2d)
        std_val = np.std(patch_2d)
        return std_val / (mean_val + 1e-8)
    
    @staticmethod
    def calculate_edge_density(patch: np.ndarray) -> float:
        """Calcula densidade de bordas usando gradiente"""
        if len(patch.shape) == 3:
            patch_2d = patch[:, :, 0]
        else:
            patch_2d = patch
            
        grad_x = np.gradient(patch_2d, axis=0)
        grad_y = np.gradient(patch_2d, axis=1)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.mean(gradient_magnitude)
    
    @classmethod
    def calculate_quality_score(cls, patch: np.ndarray) -> float:
        """Calcula score de qualidade combinado"""
        try:
            texture = cls.calculate_texture_energy(patch)
            speckle = cls.calculate_speckle_index(patch)
            edges = cls.calculate_edge_density(patch)
            
            # Normalizar e combinar métricas
            # Score mais alto = melhor qualidade
            quality_score = (texture * 0.4 + edges * 0.4 + min(speckle, 1.0) * 0.2)
            return quality_score
        except Exception:
            return 0.0


class DataNormalizer:
    """Diferentes métodos de normalização para dados SAR"""
    
    @staticmethod
    def normalize_log1p(data: np.ndarray) -> np.ndarray:
        """Normalização logarítmica (padrão para SAR)"""
        return np.log1p(np.clip(data, 0, None))
    
    @staticmethod
    def normalize_zscore(data: np.ndarray) -> np.ndarray:
        """Normalização Z-score"""
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-8)
    
    @staticmethod
    def normalize_minmax(data: np.ndarray) -> np.ndarray:
        """Normalização Min-Max"""
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val + 1e-8)
    
    @staticmethod
    def normalize_none(data: np.ndarray) -> np.ndarray:
        """Sem normalização"""
        return data
    
    @classmethod
    def normalize(cls, data: np.ndarray, method: str) -> np.ndarray:
        """Aplica normalização baseada no método especificado"""
        normalizers = {
            'log1p': cls.normalize_log1p,
            'zscore': cls.normalize_zscore,
            'minmax': cls.normalize_minmax,
            'none': cls.normalize_none
        }
        
        if method not in normalizers:
            raise ValueError(f"Método de normalização '{method}' não suportado. "
                           f"Opções: {list(normalizers.keys())}")
        
        return normalizers[method](data)


class SARPatchCreator:
    """Classe principal para criação de patches SAR"""
    
    def __init__(self, config: PatchConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.patch_metadata = []
        
    def _setup_logging(self) -> logging.Logger:
        """Configura logging com timestamp único"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f'patch_creation_log_{timestamp}.txt'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Iniciando criação de patches com configuração: {self.config}")
        return logger
    
    def inspect_image(self, image_path: str) -> Dict:
        """Inspeciona imagem e retorna informações detalhadas"""
        try:
            with rasterio.open(image_path) as src:
                image_info = {
                    'path': image_path,
                    'height': src.height,
                    'width': src.width,
                    'channels': src.count,
                    'dtype': str(src.dtypes[0]),
                    'crs': str(src.crs),
                    'transform': src.transform,
                }
                
                # Ler amostra para estatísticas
                sample_size = min(100, src.height, src.width)
                sample = src.read(window=((0, sample_size), (0, sample_size)))
                
                image_info.update({
                    'min_value': float(np.nanmin(sample)),
                    'max_value': float(np.nanmax(sample)),
                    'mean_value': float(np.nanmean(sample)),
                    'has_nan': bool(np.any(np.isnan(sample))),
                    'has_zeros': bool(np.any(sample == 0)),
                    'zero_ratio': float(np.sum(sample == 0) / sample.size)
                })
                
                self.logger.info(
                    f"Imagem {image_path}: {image_info['height']}x{image_info['width']}, "
                    f"{image_info['channels']} bandas, NaN: {image_info['has_nan']}, "
                    f"Zeros: {image_info['zero_ratio']:.2%}"
                )
                
                return image_info
                
        except Exception as e:
            self.logger.error(f'Erro ao inspecionar {image_path}: {str(e)}')
            raise
    
    def _is_valid_patch(self, patch: np.ndarray, i: int, j: int) -> Tuple[bool, str, float]:
        """Valida se um patch é adequado para treinamento"""
        # Verificar NaN
        if self.config.check_nan and np.any(np.isnan(patch)):
            return False, "NaN detectado", 0.0
        
        # Verificar pixels inválidos (zeros/extremos)
        total_pixels = patch.size
        zero_pixels = np.sum(patch == 0)
        zero_ratio = zero_pixels / total_pixels
        
        if zero_ratio > self.config.invalid_pixel_threshold:
            return False, f"Muitos zeros ({zero_ratio:.2%})", 0.0
        
        # Calcular qualidade do patch
        quality_score = PatchQualityMetrics.calculate_quality_score(patch)
        
        if quality_score < self.config.quality_threshold:
            return False, f"Baixa qualidade ({quality_score:.4f})", quality_score
        
        return True, "Válido", quality_score
    
    def _create_single_patch(self, args: Tuple) -> Optional[Dict]:
        """Cria um único patch (para paralelização)"""
        image_data, i, j, patch_size, temp_dir, image_path = args
        
        try:
            # Extrair patch
            patch = image_data[i:i+patch_size, j:j+patch_size, :]
            
            # Validar patch
            is_valid, reason, quality_score = self._is_valid_patch(patch, i, j)
            
            if not is_valid:
                return {
                    'status': 'discarded',
                    'reason': reason,
                    'position': (i, j),
                    'quality_score': quality_score
                }
            
            # Salvar patch
            patch_filename = f'patch_{i}_{j}.npy'
            patch_path = os.path.join(temp_dir, patch_filename)
            np.save(patch_path, patch)
            
            # Metadados do patch
            patch_metadata = {
                'status': 'created',
                'path': patch_path,
                'position': (i, j),
                'shape': patch.shape,
                'quality_score': quality_score,
                'mean_value': float(np.mean(patch)),
                'std_value': float(np.std(patch)),
                'source_image': image_path
            }
            
            return patch_metadata
            
        except Exception as e:
            self.logger.error(f"Erro ao criar patch ({i}, {j}): {str(e)}")
            return {
                'status': 'error',
                'position': (i, j),
                'error': str(e)
            }
    
    def create_patches_from_image(self, image_path: str) -> List[str]:
        """Cria patches de uma imagem com paralelização"""
        # Inspecionar imagem
        image_info = self.inspect_image(image_path)
        height, width = image_info['height'], image_info['width']
        
        # Verificar se imagem é menor que patch_size
        if height < self.config.patch_size or width < self.config.patch_size:
            return self._handle_small_image(image_path, image_info)
        
        # Criar diretório temporário
        temp_dir = os.path.join(self.config.output_dir, 'temp_patches')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Carregar imagem
        try:
            with rasterio.open(image_path) as src:
                image_data = src.read()  # Shape: [C, H, W]
                image_data = np.transpose(image_data, (1, 2, 0))  # Para [H, W, C]
        except Exception as e:
            self.logger.error(f'Erro ao carregar {image_path}: {str(e)}')
            raise
        
        # Aplicar normalização
        image_data = DataNormalizer.normalize(image_data, self.config.normalization)
        
        # Preparar argumentos para paralelização
        patch_args = []
        positions = []
        
        for i in range(0, height - self.config.patch_size + 1, self.config.stride):
            for j in range(0, width - self.config.patch_size + 1, self.config.stride):
                positions.append((i, j))
                patch_args.append((
                    image_data, i, j, self.config.patch_size, 
                    temp_dir, image_path
                ))
        
        self.logger.info(f"Criando {len(patch_args)} patches com {self.config.max_workers} workers")
        
        # Processar patches em paralelo
        patch_results = []
        created_patches = []
        discarded_count = 0
        error_count = 0
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submeter tarefas
            future_to_args = {
                executor.submit(self._create_single_patch, args): args 
                for args in patch_args
            }
            
            # Coletar resultados com barra de progresso
            for future in tqdm(as_completed(future_to_args), 
                             total=len(patch_args), 
                             desc="Criando patches"):
                try:
                    result = future.result()
                    if result:
                        patch_results.append(result)
                        
                        if result['status'] == 'created':
                            created_patches.append(result['path'])
                        elif result['status'] == 'discarded':
                            discarded_count += 1
                        elif result['status'] == 'error':
                            error_count += 1
                            
                except Exception as e:
                    self.logger.error(f"Erro no processamento paralelo: {str(e)}")
                    error_count += 1
        
        # Salvar metadados se configurado
        if self.config.save_metadata:
            self._save_patch_metadata(patch_results, image_path)
        
        # Log estatísticas
        self.logger.info(
            f"Imagem {image_path}: {len(created_patches)} patches criados, "
            f"{discarded_count} descartados, {error_count} erros"
        )
        
        # Estatísticas de qualidade
        quality_scores = [r['quality_score'] for r in patch_results 
                         if r['status'] == 'created']
        if quality_scores:
            self.logger.info(
                f"Qualidade dos patches - Média: {np.mean(quality_scores):.4f}, "
                f"Min: {np.min(quality_scores):.4f}, Max: {np.max(quality_scores):.4f}"
            )
        
        return created_patches
    
    def _handle_small_image(self, image_path: str, image_info: Dict) -> List[str]:
        """Processa imagem menor que patch_size como patch único"""
        self.logger.warning(
            f'Imagem pequena ({image_info["height"]}x{image_info["width"]}). '
            f'Usando como patch único.'
        )
        
        try:
            temp_dir = os.path.join(self.config.output_dir, 'temp_patches')
            os.makedirs(temp_dir, exist_ok=True)
            
            with rasterio.open(image_path) as src:
                image_data = src.read()  # Shape: [C, H, W]
                image_data = np.transpose(image_data, (1, 2, 0))  # Para [H, W, C]
            
            # Aplicar normalização
            image_data = DataNormalizer.normalize(image_data, self.config.normalization)
            
            # Validar imagem como patch único
            is_valid, reason, quality_score = self._is_valid_patch(image_data, 0, 0)
            
            if not is_valid:
                self.logger.error(f'Imagem não pode ser usada como patch: {reason}')
                return []
            
            # Salvar como patch único
            patch_path = os.path.join(temp_dir, 'patch_0_0.npy')
            np.save(patch_path, image_data)
            
            self.logger.info(f'Imagem salva como patch único: {patch_path}')
            return [patch_path]
            
        except Exception as e:
            self.logger.error(f'Erro ao processar imagem pequena: {str(e)}')
            raise
    
    def _save_patch_metadata(self, patch_results: List[Dict], image_path: str):
        """Salva metadados dos patches"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_file = f'patch_metadata_{timestamp}.yaml'
        
        metadata = {
            'source_image': image_path,
            'config': self.config.__dict__,
            'creation_time': datetime.now().isoformat(),
            'patches': patch_results,
            'statistics': {
                'total_patches': len(patch_results),
                'created': len([p for p in patch_results if p['status'] == 'created']),
                'discarded': len([p for p in patch_results if p['status'] == 'discarded']),
                'errors': len([p for p in patch_results if p['status'] == 'error'])
            }
        }
        
        try:
            with open(metadata_file, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
            self.logger.info(f"Metadados salvos em {metadata_file}")
        except Exception as e:
            self.logger.warning(f"Erro ao salvar metadados: {str(e)}")
    
    def split_patches(self, patch_paths: List[str]) -> Dict[str, List[str]]:
        """Divide patches em train/val/test"""
        if not patch_paths:
            raise ValueError('Nenhum patch disponível para divisão.')
        
        self.logger.info(f"Dividindo {len(patch_paths)} patches")
        
        # Dividir patches
        train_patches, temp_patches = train_test_split(
            patch_paths, 
            train_size=self.config.split_ratios['train'], 
            random_state=self.config.seed
        )
        
        val_size = self.config.split_ratios['val'] / (
            self.config.split_ratios['val'] + self.config.split_ratios['test']
        )
        
        val_patches, test_patches = train_test_split(
            temp_patches, 
            train_size=val_size, 
            random_state=self.config.seed
        )
        
        # Mover patches para diretórios finais
        splits = {
            'train': train_patches,
            'val': val_patches, 
            'test': test_patches
        }
        
        for split_name, patches in splits.items():
            split_dir = os.path.join(self.config.output_dir, split_name, 'images')
            os.makedirs(split_dir, exist_ok=True)
            
            for patch_path in tqdm(patches, desc=f"Movendo patches para {split_name}"):
                patch_name = os.path.basename(patch_path)
                new_name = f'{patch_name[:-4]}_{split_name}.npy'
                new_path = os.path.join(split_dir, new_name)
                
                try:
                    shutil.move(patch_path, new_path)
                except Exception as e:
                    self.logger.error(f"Erro ao mover {patch_path}: {str(e)}")
            
            self.logger.info(f'{len(patches)} patches movidos para {split_dir}')
        
        return splits
    
    def process_single_image(self, input_dir: Optional[str] = None) -> Dict:
        """Processa um único arquivo .tif"""
        input_dir = input_dir or self.config.input_dir
        
        # Listar arquivos .tif
        tif_files = list(Path(input_dir).glob('*.tif'))
        
        if not tif_files:
            raise ValueError(f'Nenhuma imagem .tif encontrada em {input_dir}')
        
        if len(tif_files) != 1:
            self.logger.warning(
                f'Encontrados {len(tif_files)} arquivos .tif. '
                f'Processando apenas o primeiro: {tif_files[0]}'
            )
        
        image_path = str(tif_files[0])
        
        # Criar patches
        start_time = time.time()
        patch_paths = self.create_patches_from_image(image_path)
        processing_time = time.time() - start_time
        
        if not patch_paths:
            raise ValueError('Nenhum patch válido foi criado')
        
        # Dividir patches ou mover para split padrão
        if self.config.split_ratios:
            splits = self.split_patches(patch_paths)
        else:
            # Mover todos para split padrão
            split_dir = os.path.join(self.config.output_dir, self.config.default_split, 'images')
            os.makedirs(split_dir, exist_ok=True)
            
            for patch_path in patch_paths:
                patch_name = os.path.basename(patch_path)
                new_name = f'{patch_name[:-4]}_{self.config.default_split}.npy'
                new_path = os.path.join(split_dir, new_name)
                shutil.move(patch_path, new_path)
            
            splits = {self.config.default_split: patch_paths}
            self.logger.info(f'{len(patch_paths)} patches movidos para {split_dir}')
        
        # Limpar diretório temporário
        temp_dir = os.path.join(self.config.output_dir, 'temp_patches')
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except OSError:
                self.logger.warning(f"Diretório temporário não pôde ser removido: {temp_dir}")
        
        # Estatísticas finais
        total_patches = sum(len(patches) for patches in splits.values())
        
        result = {
            'source_image': image_path,
            'total_patches': total_patches,
            'processing_time': processing_time,
            'splits': {k: len(v) for k, v in splits.items()},
            'patches_per_second': total_patches / processing_time if processing_time > 0 else 0
        }
        
        self.logger.info(
            f"Processamento concluído: {total_patches} patches em {processing_time:.2f}s "
            f"({result['patches_per_second']:.2f} patches/s)"
        )
        
        return result


def load_config_from_file(config_path: str) -> PatchConfig:
    """Carrega configuração de arquivo YAML"""
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return PatchConfig(**config_dict)
    except Exception as e:
        logging.error(f"Erro ao carregar configuração de {config_path}: {str(e)}")
        raise


def create_default_config_file(config_path: str = 'patch_config.yaml'):
    """Cria arquivo de configuração padrão"""
    default_config = PatchConfig()
    
    config_dict = {
        'patch_size': default_config.patch_size,
        'overlap': default_config.overlap,
        'input_dir': default_config.input_dir,
        'output_dir': default_config.output_dir,
        'split_ratios': default_config.split_ratios,
        'seed': default_config.seed,
        'expected_channels': default_config.expected_channels,
        'default_split': default_config.default_split,
        'invalid_pixel_threshold': default_config.invalid_pixel_threshold,
        'check_nan': default_config.check_nan,
        'normalization': default_config.normalization,
        'quality_threshold': default_config.quality_threshold,
        'max_workers': default_config.max_workers,
        'save_metadata': default_config.save_metadata
    }
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        print(f"Arquivo de configuração padrão criado: {config_path}")
    except Exception as e:
        print(f"Erro ao criar arquivo de configuração: {str(e)}")


def main():
    """Função principal com interface de linha de comando"""
    parser = argparse.ArgumentParser(
        description='Script melhorado para criação de patches SAR'
    )
    parser.add_argument(
        '--config', '-c', 
        type=str, 
        help='Caminho para arquivo de configuração YAML'
    )
    parser.add_argument(
        '--create-config', 
        action='store_true',
        help='Criar arquivo de configuração padrão'
    )
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        help='Diretório de entrada (sobrescreve configuração)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Diretório de saída (sobrescreve configuração)'
    )
    parser.add_argument(
        '--patch-size', '-s',
        type=int,
        help='Tamanho do patch (sobrescreve configuração)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        help='Número de workers paralelos (sobrescreve configuração)'
    )
    
    args = parser.parse_args()
    
    # Criar arquivo de configuração se solicitado
    if args.create_config:
        create_default_config_file()
        return
    
    try:
        # Carregar configuração
        if args.config:
            config = load_config_from_file(args.config)
        else:
            config = PatchConfig()
        
        # Sobrescrever configurações via argumentos
        if args.input_dir:
            config.input_dir = args.input_dir
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.patch_size:
            config.patch_size = args.patch_size
            config.stride = int(config.patch_size * (1 - config.overlap))
        if args.workers:
            config.max_workers = args.workers
        
        # Criar e executar processador
        processor = SARPatchCreator(config)
        result = processor.process_single_image()
        
        print("\n" + "="*60)
        print("PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
        print("="*60)
        print(f"Imagem processada: {result['source_image']}")
        print(f"Total de patches: {result['total_patches']}")
        print(f"Tempo de processamento: {result['processing_time']:.2f}s")
        print(f"Velocidade: {result['patches_per_second']:.2f} patches/s")
        print(f"Divisão: {result['splits']}")
        print(f"Verifique os logs para mais detalhes.")
        print("="*60)
        
    except Exception as e:
        print(f"ERRO: {str(e)}")
        logging.error(f"Erro na execução: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
