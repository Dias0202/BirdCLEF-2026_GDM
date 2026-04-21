import os
from dotenv import load_dotenv

# Mapeia o caminho absoluto do arquivo .env com base na localização do script
caminho_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(caminho_env)

# Validação para interromper a execução caso a credencial não seja lida
if not os.environ.get("KAGGLE_API_TOKEN"):
    raise ValueError("A variável KAGGLE_API_TOKEN não foi encontrada. Verifique a extensão e o local do arquivo .env.")

# Importação feita após a garantia de que as variáveis estão no ambiente
import kagglehub

path = kagglehub.competition_download('birdclef-2026')
print("Path to competition files:", path)