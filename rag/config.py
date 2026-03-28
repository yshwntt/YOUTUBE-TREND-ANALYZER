from pathlib import Path
import os
from dotenv import load_dotenv

# loading environment variables from .env file
load_dotenv()

# base project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# data folders
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

# vector db folder
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "chroma_db"

# api key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# model names
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

# chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200