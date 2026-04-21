import pandas as pd
from config import TRAIN_CSV_PATH, TAXONOMY_CSV_PATH

def run_eda():
    print("=== Fase 1: Análise Exploratória de Dados (EDA) ===")
    
    try:
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        taxonomy_df = pd.read_csv(TAXONOMY_CSV_PATH)
        
        print(f"-> train.csv carregado com sucesso! Shape: {train_df.shape}")
        print(f"-> taxonomy.csv carregado com sucesso! Shape: {taxonomy_df.shape}")
        
        print("\n[Amostra de train_df]")
        print(train_df.head(2).to_string())
        
        print("\n[Amostra de taxonomy_df]")
        print(taxonomy_df.head(2).to_string())
        
        # Checking unique classes
        num_classes = train_df['primary_label'].nunique()
        print(f"\n-> Número de classes (primary_label) únicas em train.csv: {num_classes}")
        
        if num_classes == len(taxonomy_df):
            print("-> A quantidade de classes em train.csv coincide com o taxonomy.csv.")
        else:
            print("-> AVISO: A quantidade de classes difere.")
            
        return train_df, taxonomy_df
        
    except FileNotFoundError as e:
        print(f"ERRO: Arquivo não encontrado - {e}")
        return None, None

if __name__ == "__main__":
    run_eda()
