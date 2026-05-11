import os
from dotenv import load_dotenv

class Settings:
    def __init__(self):
        try:
            load_dotenv()
            self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
            self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
            self.CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")
            
            if not self.GOOGLE_API_KEY:
                print("Warning: GOOGLE_API_KEY not found in environment variables.")
        except Exception as e:
            print(f"Error loading settings: {e}")

    @property
    def is_configured(self):
        return bool(self.GOOGLE_API_KEY)

settings = Settings()