import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEXES_DIR = DATA_DIR / "indexes"

# Create directories if not exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INDEXES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configurations
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_DIM = 384  # For all-MiniLM-L6-v2

# LLM Configuration
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")  # 'openai' or 'ollama'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Retrieval Configuration
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_ALPHA = float(os.getenv("DEFAULT_ALPHA", "0.7"))  # Semantic weight

# Classification Categories
COMPLAINT_CATEGORIES = {
    "Billing": ["charge", "invoice", "payment", "refund", "price", "bill", "subscription"],
    "Technical": ["app", "website", "crash", "bug", "error", "freeze", "load", "connection"],
    "Shipping": ["delivery", "package", "ship", "track", "arrive", "damaged", "return"],
    "Customer Service": ["support", "agent", "hold", "rude", "unhelpful", "wait", "response"],
    "Product Quality": ["defective", "quality", "broken", "poor", "material", "stitching", "work"],
    "Account": ["login", "password", "account", "profile", "access", "reset", "data"],
    "Other": []
}

# Cache settings
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour