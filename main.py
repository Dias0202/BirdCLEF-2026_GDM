import os
from dotenv import load_dotenv

# Map the absolute path of the .env file based on the script's location
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)

# Validation to interrupt execution if the credential is not read
if not os.environ.get("KAGGLE_API_TOKEN"):
    raise ValueError("The KAGGLE_API_TOKEN variable was not found. Check the extension and location of the .env file.")

# Import made after ensuring the variables are in the environment
import kagglehub

path = kagglehub.competition_download('birdclef-2026')
print("Path to competition files:", path)