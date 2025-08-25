import rasterio
import numpy as np

def analyze_sar_image(image_path):
    """Analisa a estrutura da imagem SAR"""
    with rasterio.open(image_path) as src:
        print(f"Arquivo: {image_path}")
        print(f"Dimensões: {src.height} x {src.width}")
        print(f"Número de bandas: {src.count}")
        print(f"Tipo de dados: {src.dtypes[0]}")
        print(f"CRS: {src.crs}")
        print(f"Transform: {src.transform}")
        
        # Ler uma pequena amostra para análise
        sample = src.read(window=((0, min(100, src.height)), (0, min(100, src.width))))
        print(f"Shape da amostra: {sample.shape}")
        print(f"Min: {np.nanmin(sample)}, Max: {np.nanmax(sample)}")
        print(f"Tem NaN: {np.any(np.isnan(sample))}")
        print(f"Tem zeros: {np.any(sample == 0)}")
        print(f"Percentual de zeros: {np.sum(sample == 0) / sample.size * 100:.2f}%")
        
        # Verificar estatísticas por banda
        for i in range(min(5, src.count)):  # Primeiras 5 bandas
            band_data = src.read(i + 1, window=((0, min(100, src.height)), (0, min(100, src.width))))
            print(f"Banda {i+1}: min={np.nanmin(band_data):.4f}, max={np.nanmax(band_data):.4f}, mean={np.nanmean(band_data):.4f}")

if __name__ == "__main__":
    analyze_sar_image("data/raw_images/Sentinel_1_ROI_32.tif")
