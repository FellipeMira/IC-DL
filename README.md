# Implementação de um Autoencoder Convolucional para Detecção de Mudanças em Cobertura do Solo Usando Dados SAR

Nesta seção, descrevemos a metodologia adotada para o desenvolvimento e aplicação de um autoencoder baseado em redes neurais convolucionais (CNNs) no contexto de mapeamento de mudanças em Land Use/Land Cover (LULC) causadas por enchentes no Sul do Brasil em maio de 2024. O foco principal reside na componente de codificação (encoder) do modelo, que é responsável pela extração de representações latentes compactas a partir de dados de Radar de Abertura Sintética (SAR) do Sentinel-1, com bandas VV e VH empilhadas temporalmente. Essa abordagem é não supervisionada, adequada à escassez de máscaras anotadas, e visa detectar anomalias por meio de erros de reconstrução.

## Arquitetura do Modelo: O Autoencoder Convolucional
O modelo adotado é um autoencoder CNN inspirado na estrutura U-Net, composto por um encoder (codificador) e um decoder (decodificador). O encoder opera como um extrator de características, comprimindo a informação espacial e temporal dos dados SAR em um espaço latente de dimensionalidade reduzida. Essa compressão permite aprender padrões "normais" dos dados pré-enchente, facilitando a detecção de desvios (mudanças) em observações pós-evento.
A arquitetura foi implementada utilizando a biblioteca PyTorch, com camadas convolucionais para preservar as propriedades espaciais inerentes às imagens SAR, que exibem texturas complexas devido ao speckle e ao backscattering polarimétrico (VV e VH). O número de canais de entrada (in_channels) é definido pelo empilhamento temporal das bandas, por exemplo, 8 canais para 4 observações temporais (cada com VV e VH).
## Funcionamento Detalhado do Encoder
O encoder é projetado para processar patches de imagens SAR de tamanho 512x512 (ou 256x256 para eficiência computacional), com sobreposição de 50% para aumentar o conjunto de dados e capturar contextos locais. Seu funcionamento segue um fluxo hierárquico de extração de características, inspirado em CNNs clássicas como VGG, mas adaptado para tarefas de reconstrução não supervisionada. A seguir, descrevemos o processo passo a passo:

### Pré-processamento de Entrada:

Os dados SAR são normalizados (ex.: escala logarítmica para mitigar o ruído speckle) e convertidos em tensores [C, H, W], onde C representa os canais empilhados (ex.: '0_VV', '0_VH', '1_VV', etc.).
Aumentação de dados via Albumentations (resize, flip horizontal, shift-scale-rotate ±10%, elastic transform, ajustes de brilho/contraste e HSV) é aplicada durante o treinamento para simular variações em condições de aquisição SAR, aumentando a robustez do modelo a poucas observações.
