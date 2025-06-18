import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataLoader:
    """Classe pour charger et explorer les données Optiver de manière efficace"""
    
    def __init__(self, data_path='./raw_data'):  # Ajuste le chemin selon ton organisation
        self.data_path = Path(data_path)
        self.book_train_path = self.data_path / 'book_train.parquet'
        self.book_test_path = self.data_path / 'book_test.parquet'
        self.trade_train_path = self.data_path / 'trade_train.parquet'
        self.trade_test_path = self.data_path / 'trade_test.parquet'
        self.train_csv_path = self.data_path / 'train.csv'
        self.test_csv_path = self.data_path / 'test.csv'
        
    def check_files_exist(self):
        """Vérifie quels fichiers sont disponibles"""
        files_status = {}
        for name, path in [
            ('book_train', self.book_train_path),
            ('book_test', self.book_test_path), 
            ('trade_train', self.trade_train_path),
            ('trade_test', self.trade_test_path),
            ('train_csv', self.train_csv_path),
            ('test_csv', self.test_csv_path)
        ]:
            files_status[name] = path.exists()
            print(f"{name:12}: {'✓' if path.exists() else '✗'} ({path})")
        return files_status
    
    def explore_parquet_structure(self, parquet_path):
        """Explore la structure d'un fichier parquet partitionné"""
        if not parquet_path.exists():
            print(f"Fichier non trouvé: {parquet_path}")
            return None
            
        print(f"\n=== Structure de {parquet_path.name} ===")
        
        try:
            # Lire les métadonnées sans charger toutes les données
            parquet_file = pq.ParquetFile(parquet_path)
            
            print(f"Nombre de row groups: {parquet_file.num_row_groups}")
            print(f"Schema: {parquet_file.schema}")
            
            # Si c'est partitionné, on peut voir les partitions
            if parquet_file.schema_arrow.pandas_metadata:
                print("Métadonnées pandas trouvées")
                
        except Exception as e:
            print(f"Erreur lors de la lecture des métadonnées: {e}")
            
        return parquet_file
    
    def load_sample_data(self, file_type='book', split='train', nrows=10000):
        """Charge un échantillon de données pour exploration"""
        
        if file_type == 'book' and split == 'train':
            file_path = self.book_train_path
        elif file_type == 'book' and split == 'test':
            file_path = self.book_test_path
        elif file_type == 'trade' and split == 'train':
            file_path = self.trade_train_path
        elif file_type == 'trade' and split == 'test':
            file_path = self.trade_test_path
        else:
            raise ValueError("file_type doit être 'book' ou 'trade', split doit être 'train' ou 'test'")
            
        if not file_path.exists():
            print(f"Fichier non trouvé: {file_path}")
            return None
            
        print(f"\n=== Chargement échantillon de {file_path.name} ===")
        
        try:
            # Charger un échantillon
            df = pd.read_parquet(file_path, engine='pyarrow')
            
            if len(df) > nrows:
                df = df.head(nrows)
                print(f"Échantillon de {nrows} lignes chargé (total: {len(df)} lignes dans le sample)")
            else:
                print(f"Toutes les {len(df)} lignes chargées")
                
            return df
            
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            return None
    
    def load_train_targets(self):
        """Charge les targets d'entraînement"""
        if not self.train_csv_path.exists():
            print(f"Fichier train.csv non trouvé: {self.train_csv_path}")
            return None
            
        print("\n=== Chargement des targets d'entraînement ===")
        df = pd.read_csv(self.train_csv_path)
        print(f"Shape: {df.shape}")
        return df
    
    def basic_exploration(self, df, data_name=""):
        """Exploration basique d'un DataFrame"""
        if df is None:
            return
            
        print(f"\n=== Exploration basique - {data_name} ===")
        print(f"Shape: {df.shape}")
        print(f"Colonnes: {list(df.columns)}")
        print(f"Types de données:")
        print(df.dtypes)
        print(f"\nPremières lignes:")
        print(df.head())
        print(f"\nValeurs manquantes:")
        print(df.isnull().sum())
        print(f"\nStatistiques descriptives:")
        print(df.describe())
        
        # Informations spécifiques selon le type de données
        if 'stock_id' in df.columns:
            print(f"\nNombre de stocks uniques: {df['stock_id'].nunique()}")
            print(f"Stock IDs: {sorted(df['stock_id'].unique())}")
            
        if 'time_id' in df.columns:
            print(f"Nombre de time_id uniques: {df['time_id'].nunique()}")
            print(f"Range time_id: {df['time_id'].min()} à {df['time_id'].max()}")
            
        if 'seconds_in_bucket' in df.columns:
            print(f"Range seconds_in_bucket: {df['seconds_in_bucket'].min()} à {df['seconds_in_bucket'].max()}")

def main():
    """Fonction principale pour l'exploration initiale"""
    print("=== EXPLORATION INITIALE DES DONNÉES OPTIVER ===\n")
    
    # Initialiser le loader
    loader = DataLoader()
    
    # 1. Vérifier quels fichiers existent
    print("1. Vérification des fichiers disponibles:")
    files_status = loader.check_files_exist()
    
    # 2. Explorer la structure des fichiers parquet
    print("\n2. Exploration de la structure des fichiers:")
    if files_status.get('book_train'):
        loader.explore_parquet_structure(loader.book_train_path)
    if files_status.get('trade_train'):
        loader.explore_parquet_structure(loader.trade_train_path)
    
    # 3. Charger et explorer les données book (échantillon)
    print("\n3. Exploration des données book:")
    book_sample = loader.load_sample_data('book', 'train', nrows=5000)
    if book_sample is not None:
        loader.basic_exploration(book_sample, "Book Train Sample")
    
    # 4. Charger et explorer les données trade (échantillon)
    print("\n4. Exploration des données trade:")
    trade_sample = loader.load_sample_data('trade', 'train', nrows=5000)
    if trade_sample is not None:
        loader.basic_exploration(trade_sample, "Trade Train Sample")
    
    # 5. Charger et explorer les targets
    print("\n5. Exploration des targets:")
    targets = loader.load_train_targets()
    if targets is not None:
        loader.basic_exploration(targets, "Train Targets")
    
    print("\n=== EXPLORATION TERMINÉE ===")
    
    return loader, book_sample, trade_sample, targets

# Exécuter l'exploration
if __name__ == "__main__":
    loader, book_data, trade_data, target_data = main()