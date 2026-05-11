from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.settings import settings

def get_embedding_model():
    try:
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is missing. Cannot initialize embedding model.")
            
        return GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            task_type="retrieval_document"
        )
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return None