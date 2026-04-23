import os
import numpy as np
import soundfile as sf
from pathlib import Path
import sys

# Add src to path so we can import config
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from config import TEST_SOUNDSCAPES_DIR, SAMPLE_RATE

def create_dummy_test_data():
    """
    Creates a dummy test directory and a fake 1-minute audio file
    to test the inference pipeline.
    """
    print("=== Creating Dummy Test Soundscapes ===")
    
    test_dir = TEST_SOUNDSCAPES_DIR
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 2 fake soundscapes (1 minute each)
    duration_sec = 60
    
    for i in range(1, 3):
        file_name = f"BC2026_Test_000{i}_S05_20250227_010002.ogg"
        file_path = test_dir / file_name
        
        # Generate random noise (white noise)
        noise = np.random.randn(SAMPLE_RATE * duration_sec).astype(np.float32) * 0.1
        
        # Save as OGG
        sf.write(file_path, noise, SAMPLE_RATE, format='OGG', subtype='VORBIS')
        print(f"-> Created {file_path}")
        
    print("-> Dummy test data creation complete!")

if __name__ == "__main__":
    create_dummy_test_data()