import os
import torch
from pathlib import Path
from dotenv import load_dotenv

# Points to the root 'DataCamp' directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load local .env
load_dotenv(ROOT_DIR / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()