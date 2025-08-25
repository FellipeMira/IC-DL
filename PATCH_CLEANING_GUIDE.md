# Script de Limpeza e Debug de Patches SAR

Este script (`debug_patches.py`) foi expandido para incluir funcionalidades avançadas de limpeza e otimização de patches SAR, identificando e removendo patches problemáticos que podem prejudicar o treinamento de modelos.

## 🛠️ Funcionalidades

### 1. **Análise de Problemas**
Identifica patches com:
- ❌ **Valores NaN ou infinitos**
- 🔢 **Excesso de zeros** (>95% dos pixels)
- 📊 **Valores idênticos** (sem variação)
- 🎯 **Outliers extremos** (>10% dos pixels com >5σ de desvio)
- 🏷️ **Qualidade muito baixa** (<0.01)
- 📏 **Range de valores anômalo** (muito grande ou extremo)
- 💾 **Arquivos corrompidos** ou malformados

### 2. **Otimização por Qualidade**
Remove patches abaixo de um threshold de qualidade específico para melhorar a qualidade geral do dataset.

### 3. **Sistema de Backup**
Cria backup automático de todos os patches removidos, permitindo recuperação se necessário.

### 4. **Análise Estatística**
Fornece estatísticas detalhadas sobre a distribuição de qualidade e tipos de problemas encontrados.

## 🚀 Uso

### Análise Básica (sem remoção)
```bash
# Analisar todos os patches em busca de problemas
python debug_patches.py --analyze-only

# Analisar com threshold de qualidade específico
python debug_patches.py --quality-threshold 2.5 --analyze-only

# Analisar diretório específico
python debug_patches.py --data-dir /caminho/para/dados --analyze-only
```

### Limpeza Automática
```bash
# Remover patches problemáticos (com confirmação)
python debug_patches.py

# Remover automaticamente sem confirmação
python debug_patches.py --auto-confirm

# Remover sem criar backup (NÃO recomendado)
python debug_patches.py --no-backup

# Otimizar qualidade removendo patches abaixo de threshold
python debug_patches.py --quality-threshold 2.8 --auto-confirm
```

### Debug de Criação (funcionalidade original)
```bash
# Executar debug original de criação de patches
python debug_patches.py --debug-creation
```

## 📊 Exemplo de Saída

```
🔍 ANÁLISE DE PATCHES PROBLEMÁTICOS
==================================================
🔍 Analisando todos os patches para identificar problemas...

📂 Analisando split 'train': 14 patches
📂 Analisando split 'val': 3 patches  
📂 Analisando split 'test': 3 patches

📊 RESUMO DA ANÁLISE:
  Total de patches analisados: 20
  Patches problemáticos: 0
  Taxa de problemas: 0.0%

🎯 ANÁLISE DE QUALIDADE (threshold: 2.5)
==================================================
  Patches abaixo do threshold: 2
  Qualidade média: 2.9008
  Qualidade mediana: 2.9619
  Range: 2.0991 - 3.2896
```

## ⚙️ Opções de Linha de Comando

| Opção | Descrição |
|-------|-----------|
| `--data-dir`, `-d` | Diretório com os dados (default: data) |
| `--analyze-only`, `-a` | Apenas analisar, não remover patches |
| `--auto-confirm`, `-y` | Confirmar automaticamente a remoção |
| `--no-backup` | Não criar backup dos patches removidos |
| `--quality-threshold`, `-q` | Threshold mínimo de qualidade para otimização |
| `--debug-creation` | Executar debug de criação de patches |

## 🎯 Recomendações de Uso

### Para Análise Inicial
```bash
python debug_patches.py --analyze-only
```
Execute primeiro para entender a qualidade do seu dataset.

### Para Limpeza Conservadora
```bash
python debug_patches.py --auto-confirm
```
Remove apenas patches claramente problemáticos.

### Para Otimização Agressiva
```bash
python debug_patches.py --quality-threshold 2.8 --auto-confirm
```
Remove patches de qualidade abaixo da média para datasets de alta qualidade.

### Para Datasets Críticos
```bash
python debug_patches.py --analyze-only --quality-threshold 3.0
```
Primeiro analise, depois decida o threshold baseado na distribuição.

## 🛡️ Sistema de Backup

Quando patches são removidos, um backup é criado automaticamente em:
```
data/backup_removed_patches/
├── train/images/          # Patches removidos do treino
├── val/images/            # Patches removidos da validação  
├── test/images/           # Patches removidos do teste
└── backup_info.yaml       # Informações do backup
```

O arquivo `backup_info.yaml` contém:
- Timestamp do backup
- Lista de todos os patches removidos
- Problemas encontrados em cada patch
- Caminhos originais e de backup

## 🔄 Recuperação de Backup

Para restaurar patches do backup:
```bash
# Restaurar todos os patches
cp -r data/backup_removed_patches/*/images/* data/

# Restaurar apenas um split específico
cp -r data/backup_removed_patches/train/images/* data/train/images/
```

## ⚠️ Cuidados Importantes

1. **Sempre faça backup** antes de executar limpeza em dados importantes
2. **Analise primeiro** com `--analyze-only` para entender o impacto
3. **Teste diferentes thresholds** para encontrar o equilíbrio ideal
4. **Verifique a distribuição** entre splits após a limpeza
5. **Documente as configurações** usadas para reprodutibilidade

## 📈 Benefícios para Treinamento

Patches limpos resultam em:
- ✅ **Convergência mais rápida** do modelo
- ✅ **Melhor generalização** (menos overfitting em dados ruins)
- ✅ **Métricas mais confiáveis** durante validação
- ✅ **Redução de instabilidade** no treinamento
- ✅ **Menor tempo de processamento** (menos dados, mas de melhor qualidade)

## 🚨 Casos de Uso Específicos

### Dataset Pequeno (<100 patches)
```bash
# Ser conservador, remover apenas problemas óbvios
python debug_patches.py --analyze-only
# Se necessário, usar threshold muito baixo
python debug_patches.py --quality-threshold 1.0
```

### Dataset Médio (100-1000 patches)
```bash
# Equilíbrio entre qualidade e quantidade
python debug_patches.py --quality-threshold 2.0 --auto-confirm
```

### Dataset Grande (>1000 patches)
```bash
# Ser mais agressivo na qualidade
python debug_patches.py --quality-threshold 2.8 --auto-confirm
```

### Dados de Produção
```bash
# Análise completa primeiro
python debug_patches.py --analyze-only --quality-threshold 3.0
# Backup explícito
cp -r data data_backup_manual
# Limpeza conservadora
python debug_patches.py
```
