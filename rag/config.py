from pathlib import Path

# base project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# data folders
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

# vector db folder
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "chroma_db"

# model names (local, no API required)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "google/flan-t5-small"

# chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200