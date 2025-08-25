#!/usr/bin/env python3
"""
Versão melhorada do autoencoder SAR para detecção de mudanças
Focado em análise temporal de dados Sentinel-1 para detecção de enchentes e LULC
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from albumentations import Compose, HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast, Resize
from albumentations.pytorch import ToTensorV2
import logging
from datetime import datetime
import math

# Configurações globais melhoradas
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 128
BATCH_SIZE = 4 if torch.cuda.is_available() else 2  # Adaptativo
EPOCHS = 10  # Reduzido para teste inicial
LR = 2e-4  # Learning rate ligeiramente maior
WEIGHT_DECAY = 1e-5  # Regularização L2
NUM_WORKERS = 4 if torch.cuda.is_available() else 2

# Configuração temporal específica para SAR
IN_CHANNELS = 64  # 32 tempos x 2 bandas (VV, VH)
TEMPORAL_STEPS = 32  # Número de timestamps
SPECTRAL_BANDS = 2   # VV e VH

# Logging melhorado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class TemporalSARDataset(Dataset):
    """Dataset especializado para dados SAR temporais"""
    
    def __init__(self, root_dir, transforms=None, temporal_augment=False):
        self.root_dir = root_dir
        self.transforms = transforms
        self.temporal_augment = temporal_augment
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        
    def __len__(self):
        return len(self.files)
    
    def temporal_normalize(self, image):
        """Normalização temporal específica para dados SAR"""
        # image shape: [H, W, temporal_steps * bands]
        
        # Reshape para separar temporal e espectral: [H, W, temporal, bands]
        h, w, c = image.shape
        image = image.reshape(h, w, TEMPORAL_STEPS, SPECTRAL_BANDS)
        
        # Normalização por banda ao longo do tempo
        for band in range(SPECTRAL_BANDS):
            band_data = image[:, :, :, band]  # [H, W, temporal]
            
            # Clipping específico por banda
            if band == 0:  # VV - tipicamente mais baixo
                band_data = np.clip(band_data, -25, 5)
                band_data = (band_data + 25) / 30
            else:  # VH - tipicamente ainda mais baixo
                band_data = np.clip(band_data, -35, -5)
                band_data = (band_data + 35) / 30
                
            image[:, :, :, band] = band_data
        
        # Reshape de volta: [H, W, channels]
        image = image.reshape(h, w, c)
        return image.astype(np.float32)  # CORREÇÃO: Garantir float32
    
    def temporal_augment_data(self, image):
        """Augmentação temporal específica"""
        if not self.temporal_augment or np.random.random() > 0.5:
            return image
            
        h, w, c = image.shape
        image = image.reshape(h, w, TEMPORAL_STEPS, SPECTRAL_BANDS)
        
        # Shuffle temporal aleatório (simula diferentes padrões temporais)
        if np.random.random() > 0.7:
            temporal_order = np.random.permutation(TEMPORAL_STEPS)
            image = image[:, :, temporal_order, :]
        
        # Adicionar ruído temporal suave
        if np.random.random() > 0.8:
            noise = np.random.normal(0, 0.01, image.shape).astype(np.float32)  # CORREÇÃO: Garantir float32
            image = image + noise
            
        return image.reshape(h, w, c).astype(np.float32)  # CORREÇÃO: Garantir float32.astype(np.float32)  # CORREÇÃO: Garantir float32
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.files[idx])
        image = np.load(file_path).astype(np.float32)  # CORREÇÃO: Garantir float32
        
        # Normalização temporal com conversão de tipo
        image = self.temporal_normalize(image)
        
        # Augmentação temporal com conversão de tipo
        image = self.temporal_augment_data(image)
        
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
            # ToTensorV2() já converte para tensor e reordena as dimensões
            return image.float()  # CORREÇÃO: Garantir float32 no tensor
        else:
            # Converter para tensor e garantir float32 apenas se não houve transforms
            image = torch.from_numpy(image).float()  # CORREÇÃO: Garantir float32
            
            # Reordenar dimensões se necessário: (H, W, C) -> (C, H, W)
            if len(image.shape) == 3:
                image = image.permute(2, 0, 1)
                
            return image

class SpatialAttention(nn.Module):
    """Attention espacial para focar em áreas de mudança"""
    
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention

class TemporalAttention(nn.Module):
    """Attention temporal para focar em períodos importantes"""
    
    def __init__(self, channels, temporal_steps):
        super(TemporalAttention, self).__init__()
        self.temporal_steps = temporal_steps
        self.channels_per_step = channels // temporal_steps
        
        # Rede para calcular pesos temporais
        self.temporal_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, temporal_steps),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Calcular pesos temporais
        temporal_weights = self.temporal_fc(x)  # [batch, temporal_steps]
        
        # Reshape para aplicar pesos
        x_temporal = x.view(b, self.temporal_steps, self.channels_per_step, h, w)
        
        # Aplicar pesos temporais
        weighted = x_temporal * temporal_weights.view(b, self.temporal_steps, 1, 1, 1)
        
        return weighted.view(b, c, h, w)

class ResidualBlock(nn.Module):
    """Bloco residual para melhor propagação de gradiente"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class EnhancedSARAutoencoder(nn.Module):
    """Autoencoder melhorado para detecção de mudanças SAR"""
    
    def __init__(self, in_channels=64, temporal_steps=32):
        super(EnhancedSARAutoencoder, self).__init__()
        self.temporal_steps = temporal_steps
        
        # Encoder com skip connections
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            SpatialAttention(64)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            TemporalAttention(128, temporal_steps)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            SpatialAttention(256)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512)
        )
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck com attention
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            ResidualBlock(1024),
            nn.Dropout2d(0.3)  # Regularização
        )
        
        # Decoder com skip connections
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),  # 512 + 512 from skip
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512)
        )
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),  # 256 + 256 from skip
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256)
        )
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # 128 + 128 from skip
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128)
        )
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 64 + 64 from skip
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64)
        )
        
        # Output layer
        self.final = nn.Conv2d(64, in_channels, 1)  # 1x1 conv para output
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)           # 128x128x64
        e1_pool = self.pool1(e1)    # 64x64x64
        
        e2 = self.enc2(e1_pool)     # 64x64x128
        e2_pool = self.pool2(e2)    # 32x32x128
        
        e3 = self.enc3(e2_pool)     # 32x32x256
        e3_pool = self.pool3(e3)    # 16x16x256
        
        e4 = self.enc4(e3_pool)     # 16x16x512
        e4_pool = self.pool4(e4)    # 8x8x512
        
        # Bottleneck
        bottleneck = self.bottleneck(e4_pool)  # 8x8x1024
        
        # Decoder com skip connections
        d4 = self.up4(bottleneck)   # 16x16x512
        d4 = torch.cat([d4, e4], dim=1)  # 16x16x1024
        d4 = self.dec4(d4)          # 16x16x512
        
        d3 = self.up3(d4)           # 32x32x256
        d3 = torch.cat([d3, e3], dim=1)  # 32x32x512
        d3 = self.dec3(d3)          # 32x32x256
        
        d2 = self.up2(d3)           # 64x64x128
        d2 = torch.cat([d2, e2], dim=1)  # 64x64x256
        d2 = self.dec2(d2)          # 64x64x128
        
        d1 = self.up1(d2)           # 128x128x64
        d1 = torch.cat([d1, e1], dim=1)  # 128x128x128
        d1 = self.dec1(d1)          # 128x128x64
        
        output = self.final(d1)     # 128x128x64
        
        return output

class ChangeDetectionLoss(nn.Module):
    """Função de perda específica para detecção de mudanças"""
    
    def __init__(self, alpha=0.8, beta=0.2, temporal_steps=32):
        super(ChangeDetectionLoss, self).__init__()
        self.alpha = alpha  # Peso para reconstrução
        self.beta = beta    # Peso para consistência temporal
        self.temporal_steps = temporal_steps
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
    def temporal_consistency_loss(self, pred, target):
        """Perda para garantir consistência temporal"""
        b, c, h, w = pred.shape
        
        # Reshape para [batch, temporal, bands, h, w]
        pred_temporal = pred.view(b, self.temporal_steps, c//self.temporal_steps, h, w)
        target_temporal = target.view(b, self.temporal_steps, c//self.temporal_steps, h, w)
        
        # Calcular diferenças temporais
        pred_diff = pred_temporal[:, 1:] - pred_temporal[:, :-1]
        target_diff = target_temporal[:, 1:] - target_temporal[:, :-1]
        
        return self.l1(pred_diff, target_diff)
    
    def forward(self, pred, target):
        # Perda de reconstrução principal
        reconstruction_loss = self.mse(pred, target)
        
        # Perda de consistência temporal
        temporal_loss = self.temporal_consistency_loss(pred, target)
        
        # Perda total
        total_loss = self.alpha * reconstruction_loss + self.beta * temporal_loss
        
        return total_loss, reconstruction_loss, temporal_loss

def train_epoch_enhanced(model, dataloader, optimizer, criterion, device, epoch):
    """Função de treino melhorada com logging detalhado"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_temporal_loss = 0
    
    for batch_idx, data in enumerate(dataloader):
        data = data.to(device).float()  # CORREÇÃO: Garantir float32 no dispositivo
        optimizer.zero_grad()
        
        output = model(data)
        loss, recon_loss, temporal_loss = criterion(output, data)
        
        loss.backward()
        
        # Gradient clipping para estabilidade
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_temporal_loss += temporal_loss.item()
        
        if batch_idx % 10 == 0:
            logging.info(f'Epoch {epoch}, Batch {batch_idx}: '
                        f'Loss={loss.item():.4f}, '
                        f'Recon={recon_loss.item():.4f}, '
                        f'Temporal={temporal_loss.item():.4f}')
    
    return (total_loss / len(dataloader), 
            total_recon_loss / len(dataloader),
            total_temporal_loss / len(dataloader))

def val_epoch_enhanced(model, dataloader, criterion, device):
    """Função de validação melhorada"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_temporal_loss = 0
    
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device).float()  # CORREÇÃO: Garantir float32 no dispositivo
            output = model(data)
            loss, recon_loss, temporal_loss = criterion(output, data)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_temporal_loss += temporal_loss.item()
    
    return (total_loss / len(dataloader),
            total_recon_loss / len(dataloader),
            total_temporal_loss / len(dataloader))

def train_model_enhanced(model, train_loader, val_loader, epochs, lr, device, 
                        checkpoint_path='best_sar_model.pth'):
    """Função de treinamento melhorada"""
    
    # Otimizador com weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    
    # Scheduler para learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Função de perda especializada
    criterion = ChangeDetectionLoss(temporal_steps=TEMPORAL_STEPS)
    
    best_val_loss = float('inf')
    patience = 20  # Early stopping mais paciente
    counter = 0
    
    logging.info(f"Starting training with {epochs} epochs on {device}")
    
    for epoch in range(epochs):
        # Treinamento
        train_loss, train_recon, train_temporal = train_epoch_enhanced(
            model, train_loader, optimizer, criterion, device, epoch+1
        )
        
        # Validação
        val_loss, val_recon, val_temporal = val_epoch_enhanced(
            model, val_loader, criterion, device
        )
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch+1}/{epochs} - LR: {current_lr:.6f}')
        logging.info(f'Train - Total: {train_loss:.4f}, Recon: {train_recon:.4f}, Temporal: {train_temporal:.4f}')
        logging.info(f'Val - Total: {val_loss:.4f}, Recon: {val_recon:.4f}, Temporal: {val_temporal:.4f}')
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - LR: {current_lr:.6f}')
        
        # Salvar melhor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, checkpoint_path)
            logging.info(f'New best model saved with val_loss: {val_loss:.4f}')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logging.info('Early stopping triggered.')
                break

# Transforms melhorados para dados SAR temporais
enhanced_train_transforms = Compose([
    Resize(PATCH_SIZE, PATCH_SIZE),
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.03, scale_limit=0.03, rotate_limit=3, p=0.2),
    RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.2),
    ToTensorV2()
])

enhanced_val_transforms = Compose([
    Resize(PATCH_SIZE, PATCH_SIZE),
    ToTensorV2()
])

if __name__ == '__main__':
    logging.info("Initializing Enhanced SAR Change Detection Autoencoder")
    
    # Datasets com augmentação temporal
    train_dataset = TemporalSARDataset(
        'data/train/images', 
        transforms=enhanced_train_transforms,
        temporal_augment=True
    )
    val_dataset = TemporalSARDataset(
        'data/val/images', 
        transforms=enhanced_val_transforms,
        temporal_augment=False
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )
    
    # Modelo melhorado
    model = EnhancedSARAutoencoder(
        in_channels=IN_CHANNELS,
        temporal_steps=TEMPORAL_STEPS
    ).to(DEVICE).float()  # CORREÇÃO: Garantir float32 para o modelo
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    # Treinamento
    train_model_enhanced(model, train_loader, val_loader, EPOCHS, LR, DEVICE)
    
    logging.info("Training completed!")
