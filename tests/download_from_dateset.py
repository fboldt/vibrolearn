import sys
import os

# Adiciona a raiz do projeto ao path do Python para que ele encontre a pasta 'dataset'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.utils import download_dataset

def test_downloads():
    print("Iniciando testes de download dos datasets...\n")

    # cwru all
    download_dataset("dataset/cwru/config.csv", "raw_data/cwru")

    # cwru sniped
    #download_dataset("dataset/cwru/config.csv", "raw_data/cwru", filenames=["97.mat"])

    # paderborn
    # print("Testando download do Paderborn (K003.rar)...")
    # download_dataset("dataset/paderborn/config.csv", "raw_data/paderborn", filenames=["K003.rar"])
    
    # print("\nFile downloaded!!")

if __name__ == "__main__":
    test_downloads()