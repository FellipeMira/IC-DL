import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from albumentations import Compose, HorizontalFlip, ShiftScaleRotate, ElasticTransform, RandomBrightnessContrast, HueSaturationValue, Resize
from albumentations.pytorch import ToTensorV2
import logging
from datetime import datetime

# Configurações globais
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 512  # Ou 256 para testes mais rápidos
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
NUM_WORKERS = 4  # Ajuste baseado na máquina

# Logging
logging.basicConfig(filename=f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', level=logging.INFO)


class SARAutoencoderDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        """
        Dataset para patches SAR.
        - root_dir: pasta com patches .npy (ex.: train/images)
        - transforms: Albumentations para augment
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.files[idx])
        image = np.load(file_path)  # Shape: [H, W, C] (C = tempos x 2 bandas)
        image = image.astype(np.float32)  # Normalize se necessário (ex.: np.log1p(image) para SAR speckle)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']

        return image  # Retorna tensor [C, H, W] para autoencoder (input == target)

# Transforms para treino
train_transforms = Compose([
    Resize(PATCH_SIZE, PATCH_SIZE),
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
    ElasticTransform(p=0.3),
    RandomBrightnessContrast(p=0.3),
    HueSaturationValue(p=0.3),  # Adaptado para SAR (ajusta intensidades)
    ToTensorV2()
])

# Transforms para val/test (apenas resize e tensor)
val_transforms = Compose([
    Resize(PATCH_SIZE, PATCH_SIZE),
    ToTensorV2()
])

# Modelo Autoencoder CNN
class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels):
        """
        Autoencoder CNN para reconstrução de imagens SAR.
        - in_channels: número de canais no input (ex.: 8 para 4 tempos x 2 bandas)
        """
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
        self.enc4 = nn.Sequential(nn.Conv2d(256, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())  # Bottleneck
        
        # Decoder
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(32, 256, 2, stride=2), nn.BatchNorm2d(256), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),  # Skip from enc3
                                  nn.ConvTranspose2d(256, 128, 2, stride=2), nn.BatchNorm2d(128), nn.ReLU())
        self.dec3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),  # Skip from enc2
                                  nn.ConvTranspose2d(128, 64, 2, stride=2), nn.BatchNorm2d(64), nn.ReLU())
        self.dec4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),  # Skip from enc1
                                  nn.Conv2d(64, in_channels, 3, padding=1), nn.Sigmoid())  # Output reconstruído

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        latent = self.enc4(e3)
        
        # Decoder com skips
        d1 = self.dec1(latent)
        d2 = self.dec2(torch.cat([d1, e3], dim=1))
        d3 = self.dec3(torch.cat([d2, e2], dim=1))
        output = self.dec4(torch.cat([d3, e1], dim=1))
        return output

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)  # Reconstrução: input == target
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def val_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_model(model, train_loader, val_loader, epochs, lr, device, checkpoint_path='best_model.pth'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience = 5  # Early stopping opcional
    counter = 0
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = val_epoch(model, val_loader, criterion, device)
        
        logging.info(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
        
        # Salva melhor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping triggered.')
                break

if __name__ == '__main__':
    # Defina in_channels baseado no seu cubo (ex.: len(['0_VV', '0_VH', '1_VV', '1_VH', ...]) )
    IN_CHANNELS = 8  # 4 tempos x 2 bandas

    # Datasets e Loaders
    train_dataset = SARAutoencoderDataset('data/train/images', transforms=train_transforms)
    val_dataset = SARAutoencoderDataset('data/val/images', transforms=val_transforms)
    test_dataset = SARAutoencoderDataset('data/test/images', transforms=val_transforms)  # Para inferência posterior

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)  # Batch 1 para visualização

    # Modelo
    model = ConvAutoencoder(IN_CHANNELS).to(DEVICE)

    # Treinamento
    train_model(model, train_loader, val_loader, EPOCHS, LR, DEVICE)
