#!/usr/bin/env python3
"""
Exemplo de uso do sistema de criação de patches SAR melhorado.
Este script demonstra como usar as principais funcionalidades.
"""

import os
import sys
from pathlib import Path

# Adicionar diretório atual ao path para importar módulos locais
sys.path.append(str(Path(__file__).parent))

from patch_data_improved import SARPatchCreator, PatchConfig
from visualize_patches import PatchVisualizer


def exemplo_basico():
    """Exemplo básico de criação de patches"""
    print("="*60)
    print("EXEMPLO 1: Criação Básica de Patches")
    print("="*60)
    
    # Configuração básica
    config = PatchConfig(
        patch_size=128,
        overlap=0.3,
        input_dir='data/raw_images',
        output_dir='data',
        normalization='none',
        max_workers=2
    )
    
    # Criar processador
    processor = SARPatchCreator(config)
    
    try:
        # Processar imagem
        result = processor.process_single_image()
        
        print(f"✅ Processamento concluído!")
        print(f"📊 Total de patches: {result['total_patches']}")
        print(f"⏱️ Tempo: {result['processing_time']:.2f}s")
        print(f"📈 Velocidade: {result['patches_per_second']:.2f} patches/s")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return False


def exemplo_configuracao_customizada():
    """Exemplo com configuração customizada"""
    print("\n" + "="*60)
    print("EXEMPLO 2: Configuração Customizada")
    print("="*60)
    
    # Configuração para patches maiores com alta qualidade
    config = PatchConfig(
        patch_size=256,
        overlap=0.5,  # Maior sobreposição
        normalization='log1p',  # Normalização logarítmica para SAR
        quality_threshold=0.2,  # Maior threshold de qualidade
        invalid_pixel_threshold=0.90,  # Mais restritivo
        max_workers=4,
        save_metadata=True
    )
    
    processor = SARPatchCreator(config)
    
    try:
        result = processor.process_single_image()
        
        print(f"✅ Patches de alta qualidade criados!")
        print(f"📊 Total: {result['total_patches']}")
        print(f"📋 Configuração salva nos metadados")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return False


def exemplo_analise_patches():
    """Exemplo de análise dos patches criados"""
    print("\n" + "="*60)
    print("EXEMPLO 3: Análise de Patches")
    print("="*60)
    
    # Criar visualizador
    visualizer = PatchVisualizer('data')
    
    # Verificar patches disponíveis
    total_patches = sum(len(patches) for patches in visualizer.patches_info.values())
    
    if total_patches == 0:
        print("❌ Nenhum patch encontrado. Execute os exemplos anteriores primeiro.")
        return False
    
    print(f"📊 Encontrados {total_patches} patches:")
    for split, patches in visualizer.patches_info.items():
        if patches:
            print(f"   {split}: {len(patches)} patches")
    
    # Analisar split de treino
    if visualizer.patches_info['train']:
        print("\n🔍 Analisando patches de treino...")
        stats = visualizer.analyze_patch_statistics('train')
        
        if stats and 'global_stats' in stats:
            gs = stats['global_stats']
            print(f"   📈 Média global: {gs['mean_of_means']:.4f}")
            print(f"   📏 Forma comum: {gs['common_shape']}")
            print(f"   🎯 Qualidade média: {gs.get('mean_quality', 'N/A')}")
    
    print("✅ Análise concluída!")
    return True


def exemplo_relatorio_completo():
    """Exemplo de geração de relatório completo"""
    print("\n" + "="*60)
    print("EXEMPLO 4: Relatório Completo")
    print("="*60)
    
    visualizer = PatchVisualizer('data')
    
    # Verificar se há dados
    total_patches = sum(len(patches) for patches in visualizer.patches_info.values())
    
    if total_patches == 0:
        print("❌ Nenhum patch encontrado para análise.")
        return False
    
    try:
        print("📋 Gerando relatório completo...")
        report_file = visualizer.save_analysis_report('analysis_reports')
        
        print(f"✅ Relatório gerado com sucesso!")
        print(f"📄 Arquivo principal: {report_file}")
        print(f"📁 Diretório: analysis_reports/")
        print(f"🖼️ Gráficos salvos como PNG")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao gerar relatório: {str(e)}")
        return False


def exemplo_configuracao_personalizada():
    """Exemplo de diferentes configurações para casos específicos"""
    print("\n" + "="*60)
    print("EXEMPLO 5: Configurações Especializadas")
    print("="*60)
    
    # Configuração para prototipagem rápida
    config_rapido = PatchConfig(
        patch_size=64,
        overlap=0.0,  # Sem sobreposição = mais rápido
        quality_threshold=0.01,  # Threshold baixo = mais patches
        max_workers=8,
        save_metadata=False  # Sem metadados = mais rápido
    )
    
    # Configuração para produção de alta qualidade
    config_qualidade = PatchConfig(
        patch_size=512,
        overlap=0.7,  # Alta sobreposição = mais dados
        normalization='log1p',
        quality_threshold=0.5,  # Alta qualidade
        invalid_pixel_threshold=0.80,  # Mais restritivo
        max_workers=2,  # Menos workers para patches grandes
        save_metadata=True
    )
    
    # Configuração para dados limitados
    config_limitado = PatchConfig(
        patch_size=128,
        overlap=0.8,  # Máxima sobreposição
        quality_threshold=0.01,  # Aceitar qualquer qualidade
        invalid_pixel_threshold=0.99,  # Muito permissivo
        normalization='minmax'
    )
    
    configs = {
        'Prototipagem Rápida': config_rapido,
        'Alta Qualidade': config_qualidade,
        'Dados Limitados': config_limitado
    }
    
    print("🎛️ Configurações disponíveis:")
    for nome, config in configs.items():
        print(f"\n📋 {nome}:")
        print(f"   Patch size: {config.patch_size}")
        print(f"   Overlap: {config.overlap}")
        print(f"   Quality threshold: {config.quality_threshold}")
        print(f"   Normalização: {config.normalization}")
        print(f"   Workers: {config.max_workers}")
    
    print("\n💡 Dica: Copie essas configurações para seu arquivo YAML")
    return True


def main():
    """Função principal que executa todos os exemplos"""
    print("🚀 EXEMPLOS DO SISTEMA DE PATCHES SAR MELHORADO")
    print("Este script demonstra as principais funcionalidades.")
    
    # Verificar se a imagem de entrada existe
    input_image = Path('data/raw_images/Sentinel_1_ROI_32.tif')
    if not input_image.exists():
        print(f"\n❌ Imagem de entrada não encontrada: {input_image}")
        print("   Coloque sua imagem SAR em data/raw_images/")
        return 1
    
    print(f"\n✅ Imagem encontrada: {input_image}")
    
    # Executar exemplos sequencialmente
    exemplos = [
        ("Criação Básica", exemplo_basico),
        ("Configuração Customizada", exemplo_configuracao_customizada),
        ("Análise de Patches", exemplo_analise_patches),
        ("Relatório Completo", exemplo_relatorio_completo),
        ("Configurações Especializadas", exemplo_configuracao_personalizada)
    ]
    
    sucesso = 0
    for nome, funcao in exemplos:
        try:
            if funcao():
                sucesso += 1
                print(f"✅ {nome}: Concluído")
            else:
                print(f"⚠️ {nome}: Completado com avisos")
        except KeyboardInterrupt:
            print(f"\n⚠️ Interrompido pelo usuário em: {nome}")
            break
        except Exception as e:
            print(f"❌ {nome}: Erro - {str(e)}")
    
    print(f"\n🎯 RESUMO: {sucesso}/{len(exemplos)} exemplos executados com sucesso")
    
    if sucesso > 0:
        print("\n📁 Arquivos gerados:")
        print("   data/train/images/ - Patches de treinamento")
        print("   data/val/images/ - Patches de validação")
        print("   data/test/images/ - Patches de teste")
        print("   analysis_reports/ - Relatórios e visualizações")
        print("   *.log - Logs de execução")
        print("   *.yaml - Metadados e configurações")
    
    print("\n🔗 Próximos passos:")
    print("   1. Examine os patches gerados em data/")
    print("   2. Verifique os relatórios em analysis_reports/")
    print("   3. Ajuste configurações em patch_config.yaml")
    print("   4. Use os patches em seu modelo de machine learning")
    
    return 0


if __name__ == '__main__':
    exit(main())
