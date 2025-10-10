import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")  # Replace with strong env value in production
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

# AI settings
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")   # default AI provider
AI_MODEL = os.getenv("AI_MODEL", "gpt-4")          # default AI model
AI_API_KEY = os.getenv("AI_API_KEY", "")           # API key for AI provider

# App paths
REGION_PATH = os.getenv("REGION_PATH", "app/api/regions")  # path to region JSON files
