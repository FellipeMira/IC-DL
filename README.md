# 🛰️ Enhanced SAR Autoencoder para Detecção de Mudanças

## 📋 Visão Geral

Este projeto implementa um **autoencoder U-Net** especializado para detecção de mudanças em imagens SAR (Synthetic Aperture Radar) multi-temporais do Sentinel-1. A rede neural utiliza mecanismos de atenção temporal e espacial para identificar mudanças na superfície terrestre, como enchentes, desmatamento e alterações de uso do solo.

A rede Enhanced SAR Autoencoder é um U‑Net adaptado para séries temporais Sentinel‑1, com 64 canais de entrada (32 épocas × 2 bandas VV/VH) definidos globalmente para lidar com a dimensão temporal e espectral do SAR. O encoder empilha blocos residuais e mecanismos de atenção: o primeiro bloco aplica atenção espacial para destacar regiões relevantes no mapa (Conv2D 7×7 + sigmoid), enquanto o segundo utiliza atenção temporal ao projetar cada canal em pesos normalizados via softmax, realçando períodos críticos da série. Esses módulos são integrados ao U‑Net com skip connections, formando um autoencoder profundo que comprime as features no bottleneck e as reconstrói simetricamente no decoder.

Durante o treinamento, a função ChangeDetectionLoss combina erro de reconstrução (MSE) e consistência temporal (L1 entre diferenças consecutivas), pressionando a rede a aprender padrões “normais” e manter coerência ao longo do tempo. Na fase de inferência, o analisador calcula o erro absoluto entre entrada e reconstrução para cada pixel; um limiar baseado em percentil (por padrão, 95%) gera um mapa binário de mudanças, com pós-processamento morfológico para reduzir ruídos. Assim, regiões com alto erro indicam alterações significativas na série temporal SAR, como enchentes ou desmatamentos, sendo realçadas pela atenção espacial e temporal que foca nos locais e instantes mais impactados.

## 🧠 Arquitetura da Rede Neural

### 1. Estrutura do Autoencoder

A rede é baseada na arquitetura **U-Net** com as seguintes características:

```
Input: [Batch, 64, 448, 336]  → 64 canais (32 timesteps × 2 bandas SAR)
                ↓
    ┌─────────────────────────────────────┐
    │          ENCODER BRANCH             │
    │  ┌─────────────────────────────────┐ │
    │  │ Conv2D(64→64) + BatchNorm + ReLU│ │  Skip Connection 1
    │  │ ResidualBlock + SpatialAttention│ │ ─────────────┐
    │  └─────────────────────────────────┘ │              │
    │             MaxPool(2×2)             │              │
    │                 ↓                    │              │
    │  ┌─────────────────────────────────┐ │              │
    │  │Conv2D(64→128) + BatchNorm + ReLU│ │ Skip Connection 2
    │  │ResidualBlock + TemporalAttention│ │ ─────────────┼─┐
    │  └─────────────────────────────────┘ │              │ │
    │             MaxPool(2×2)             │              │ │
    │                 ↓                    │              │ │
    │  ┌─────────────────────────────────┐ │              │ │
    │  │Conv2D(128→256) + BatchNorm+ReLU │ │ Skip Connection 3
    │  │ ResidualBlock + SpatialAttention│ │ ─────────────┼─┼─┐
    │  └─────────────────────────────────┘ │              │ │ │
    │             MaxPool(2×2)             │              │ │ │
    │                 ↓                    │              │ │ │
    │  ┌─────────────────────────────────┐ │              │ │ │
    │  │Conv2D(256→512) + BatchNorm+ReLU │ │ Skip Connection 4
    │  │       ResidualBlock             │ │ ─────────────┼─┼─┼─┐
    │  └─────────────────────────────────┘ │              │ │ │ │
    │             MaxPool(2×2)             │              │ │ │ │
    └───────────────────────────────────── ┘              │ │ │ │
                        ↓                                 │ │ │ │
    ┌───────────────────────────────────── ┐              │ │ │ │
    │           BOTTLENECK                 │              │ │ │ │
    │  Conv2D(512→1024) + BatchNorm+ReLU   │              │ │ │ │
    │  ResidualBlock + Dropout(0.3)        │              │ │ │ │
    └───────────────────────────────────── ┘              │ │ │ │
                        ↓                                 │ │ │ │
    ┌───────────────────────────────────── ┐              │ │ │ │
    │          DECODER BRANCH              │              │ │ │ │
    │  ┌─────────────────────────────────┐ │              │ │ │ │
    │  │  ConvTranspose2D(1024→512)      │ │              │ │ │ │
    │  │  Concatenate with Skip 4  ──────┼─┼──────────────┘ │ │ │
    │  │  Conv2D + ResidualBlock         │ │                │ │ │
    │  └─────────────────────────────────┘ │                │ │ │
    │                 ↓                    │                │ │ │
    │  ┌─────────────────────────────────┐ │                │ │ │
    │  │  ConvTranspose2D(512→256)       │ │                │ │ │
    │  │  Concatenate with Skip 3  ──────┼─┼────────────────┘ │ │
    │  │  Conv2D + ResidualBlock         │ │                  │ │
    │  └─────────────────────────────────┘ │                  │ │
    │                 ↓                    │                  │ │
    │  ┌─────────────────────────────────┐ │                  │ │
    │  │  ConvTranspose2D(256→128)       │ │                  │ │
    │  │  Concatenate with Skip 2  ──────┼─┼──────────────────┘ │
    │  │  Conv2D + ResidualBlock         │ │                    │
    │  └─────────────────────────────────┘ │                    │
    │                 ↓                    │                    │
    │  ┌─────────────────────────────────┐ │                    │
    │  │  ConvTranspose2D(128→64)        │ │                    │
    │  │  Concatenate with Skip 1  ──────┼─┼────────────────────┘
    │  │  Conv2D + ResidualBlock         │ │
    │  └─────────────────────────────────┘ │
    │                 ↓                    │
    │  ┌─────────────────────────────────┐ │
    │  │      Conv2D(64→64)              │ │
    │  │     Output Layer                │ │
    │  └─────────────────────────────────┘ │
    └───────────────────────────────────── ┘
                        ↓
Output: [Batch, 64, 448, 336]  → Reconstrução da imagem original
```

### 2. Componentes Especializados

#### 🎯 **Mecanismo de Atenção Espacial**
```python
class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.conv(x)          # Gera mapa de atenção
        attention = self.sigmoid(attention) # Normaliza [0,1]
        return x * attention              # Aplica atenção
```

**Função:** Foca automaticamente em regiões espaciais importantes para detecção de mudanças.

#### ⏰ **Mecanismo de Atenção Temporal**
```python
class TemporalAttention(nn.Module):
    def __init__(self, channels, temporal_steps):
        super().__init__()
        self.temporal_steps = temporal_steps
        self.temporal_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, temporal_steps),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Calcula pesos para cada timestep
        temporal_weights = self.temporal_fc(x)
        # Aplica pesos aos canais temporais
        return weighted_temporal_features
```

**Função:** Identifica automaticamente quais períodos temporais são mais relevantes para detectar mudanças.

#### 🔄 **Blocos Residuais**
```python
class ResidualBlock(nn.Module):
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Conexão residual
        return self.relu(out)
```

**Função:** Facilita o treinamento de redes profundas e evita o problema do gradiente que desaparece.

## 🔍 Como a Rede Detecta Mudanças

### 1. Princípio do Autoencoder para Detecção de Mudanças

O autoencoder funciona baseado no princípio de **"aprender o normal"**:

```
┌─────────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Imagem Original   │───▶│   Autoencoder    │───▶│ Imagem Reconstruída │
│   [Todas as épocas] │    │ [Aprende padrões │    │  [Padrões normais]  │
│                     │    │     normais]     │    │                     │
└─────────────────────┘    └──────────────────┘    └─────────────────────┘
           │                                                    │
           │                ┌──────────────────┐               │
           └───────────────▶│ Erro de Reconst. │◀──────────────┘
                            │ |Original - Recon|│
                            └──────────────────┘
                                       │
                                       ▼
                            ┌──────────────────┐
                            │ Mapa de Mudanças │
                            │ (Alto erro =     │
                            │  Mudança)        │
                            └──────────────────┘
```

### 2. Processo de Detecção

#### **Etapa 1: Treinamento**
- A rede aprende a reconstruir **apenas áreas sem mudanças**
- Dados de treinamento contêm pixels "estáveis" da série temporal
- A rede memoriza padrões espectrais e temporais normais

#### **Etapa 2: Inferência**
```python
def detect_changes(original_image, model):
    # 1. Normalização temporal por banda
    normalized = normalize_temporal_bands(original_image)
    
    # 2. Reconstrução pela rede
    with torch.no_grad():
        reconstruction = model(normalized)
    
    # 3. Cálculo do erro de reconstrução
    error = torch.mean((original_image - reconstruction) ** 2, dim=1)
    
    # 4. Threshold para detecção binária
    threshold = torch.quantile(error, 0.95)  # 95° percentil
    changes = error > threshold
    
    return error, changes
```

#### **Etapa 3: Interpretação**
- **Erro baixo**: Área reconstruída corretamente → **Sem mudança**
- **Erro alto**: Área mal reconstruída → **Mudança detectada**

### 3. Tratamento Temporal

#### **Estrutura dos Dados SAR**
```
Input Tensor: [Batch, 64, Height, Width]
                 ↓
Interpretação: [Batch, 32×2, Height, Width]
                        │
                ┌───────┴───────┐
                │               │
           32 Timesteps    2 Bandas
           (Sentinel-1)    (VV, VH)
```

#### **Normalização Temporal por Banda**
```python
def temporal_normalize(image):
    # Reshape: [H, W, 64] → [H, W, 32, 2]
    reshaped = image.reshape(H, W, 32, 2)
    
    for band in range(2):
        band_data = reshaped[:, :, :, band]  # [H, W, 32]
        
        if band == 0:  # Banda VV
            # Backscatter mais forte, range típico: [-25, 5] dB
            clipped = np.clip(band_data, -25, 5)
            normalized = (clipped + 25) / 30  # [0, 1]
        else:  # Banda VH
            # Backscatter mais fraco, range típico: [-35, -5] dB
            clipped = np.clip(band_data, -35, -5) 
            normalized = (clipped + 35) / 30  # [0, 1]
        
        reshaped[:, :, :, band] = normalized
    
    return reshaped.reshape(H, W, 64)
```

## 🎯 Função de Perda Especializada

### **ChangeDetectionLoss**

A rede usa uma função de perda combinada que otimiza dois objetivos:

```python
class ChangeDetectionLoss(nn.Module):
    def forward(self, pred, target):
        # 1. Perda de Reconstrução (MSE)
        reconstruction_loss = MSE(pred, target)
        
        # 2. Perda de Consistência Temporal
        temporal_loss = temporal_consistency(pred, target)
        
        # 3. Combinação Ponderada
        total_loss = α × reconstruction_loss + β × temporal_loss
        
        return total_loss
```

#### **1. Perda de Reconstrução (α = 0.8)**
- **Objetivo**: Minimizar erro pixel-a-pixel
- **Matemática**: `MSE = mean((original - reconstructed)²)`
- **Função**: Força a rede a reconstruir fielmente áreas normais

#### **2. Perda de Consistência Temporal (β = 0.2)**
```python
def temporal_consistency_loss(pred, target):
    # Reshape: [B, 64, H, W] → [B, 32, 2, H, W]
    pred_temporal = pred.view(B, 32, 2, H, W)
    target_temporal = target.view(B, 32, 2, H, W)
    
    # Diferenças temporais consecutivas
    pred_diff = pred_temporal[:, 1:] - pred_temporal[:, :-1]
    target_diff = target_temporal[:, 1:] - target_temporal[:, :-1]
    
    # L1 loss nas diferenças temporais
    return L1(pred_diff, target_diff)
```
- **Objetivo**: Manter coerência temporal na reconstrução
- **Função**: Evita artefatos temporais e melhora estabilidade

## ⚙️ Pipeline de Processamento

### **1. Pré-processamento**
```python
# Correção de valores inválidos
image = np.nan_to_num(image, nan=median_value)

# Normalização robusta por percentis
p2, p98 = np.percentile(valid_pixels, [2, 98])
normalized = (image - p2) / (p98 - p2)
```

### **2. Arquitetura de Processamento**
```
Imagem SAR Original
        ↓
┌───────────────────┐
│ Correção de NaN   │ ← Substituição por mediana
│ e valores Inf     │
└───────────────────┘
        ↓
┌───────────────────┐
│ Redimensionamento │ ← Compatibilidade com U-Net
│ 442×330 → 448×336 │   (múltiplos de 16)
└───────────────────┘
        ↓
┌───────────────────┐
│ Normalização      │ ← Por banda (VV/VH)
│ Temporal          │   Range [0,1]
└───────────────────┘
        ↓
┌───────────────────┐
│ Inferência        │ ← Modelo U-Net + Atenção
│ Neural Network    │
└───────────────────┘
        ↓
┌───────────────────┐
│ Cálculo de Erro   │ ← MSE pixel-wise
│ de Reconstrução   │
└───────────────────┘
        ↓
┌───────────────────┐
│ Threshold         │ ← 95° percentil
│ Adaptativo        │
└───────────────────┘
        ↓
┌───────────────────┐
│ Mapa de Mudanças  │ ← Binário + Contínuo
│ Final             │
└───────────────────┘
```

### **3. Saída Georreferenciada**
```python
# Preservação de metadados geoespaciais
with rasterio.open('output.tif', 'w',
                   driver='GTiff',
                   height=original_height,
                   width=original_width,
                   count=1,
                   dtype=np.uint8,
                   crs=original_crs,        # Sistema de coordenadas
                   transform=original_transform  # Geotransformação
                  ) as dst:
    dst.write(change_map, 1)
```

## 📊 Interpretação dos Resultados

### **Métricas de Saída**
- **Erro Médio**: Indica qualidade geral da reconstrução
- **Threshold Adaptativo**: Baseado no 95° percentil dos erros
- **Porcentagem de Mudanças**: % da área total com mudanças detectadas
- **Mapa Contínuo**: Valores de erro [0, 1] para análise detalhada
- **Mapa Binário**: Mudanças binárias (0 = sem mudança, 255 = mudança)

### **Tipos de Mudanças Detectáveis**
1. **Enchentes**: Mudança drástica em backscatter por presença de água
2. **Desmatamento**: Alteração de textura e intensidade SAR
3. **Construções**: Aumento significativo de backscatter
4. **Mudanças Agrícolas**: Variações sazonais em cultivos
5. **Deslizamentos**: Alterações topográficas e de cobertura

### **Vantagens da Abordagem**
- ✅ **Não supervisionada**: Não requer dados rotulados de mudanças
- ✅ **Adaptativa**: Threshold automático baseado nos dados
- ✅ **Temporal**: Exploração completa da dimensão temporal
- ✅ **Robusta**: Mecanismos de atenção e normalização adaptativa
- ✅ **Escalável**: Processamento de imagens inteiras sem patches

## 🔬 Aspectos Técnicos Avançados

### **Attention Mechanisms**
- **Spatial Attention**: Kernel 7×7 para captura de contexto espacial
- **Temporal Attention**: Softmax sobre 32 timesteps para seleção temporal
- **Integração**: Aplicada em diferentes níveis da rede para máxima efetividade

### **Otimização e Regularização**
- **Optimizer**: AdamW com weight decay (1e-5)
- **Learning Rate**: 2e-4 com ReduceLROnPlateau scheduler
- **Gradient Clipping**: max_norm=1.0 para estabilidade
- **Dropout**: 0.3 no bottleneck para generalização
- **Early Stopping**: Paciência de 20 épocas

### **Compatibilidade de Dados**
- **Entrada**: Multi-temporal SAR (32 timesteps × 2 bandas)
- **Tipos**: Float32 para consistência numérica
- **Dimensões**: Compatível com U-Net (múltiplos de 16)
- **Formato**: GeoTIFF com preservação de CRS e geotransformação

---

**Desenvolvido com**: PyTorch 2.8.0, Rasterio 1.4.3, Albumentations 2.0.8
**Suporte**: CUDA para aceleração GPU, processamento de imagens SAR Sentinel-1
**Aplicações**: Monitoramento ambiental, detecção de desastres, análise de LULC
