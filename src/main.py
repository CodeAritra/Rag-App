from ingestion.loader import load_documents
from ingestion.splitter import split_documents
from retrieval.vector_store import create_vector_store
from config.settings import settings
import sys

def ingestion():
    print("--- RAG Ingestion Pipeline ---")
    
    if not settings.is_configured:
        print("Error: Settings not properly configured. Please check your .env file.")
        sys.exit(1)

    try:
        # 1. Ingest
        documents = load_documents()
        if not documents:
            print("No documents found. Exiting.")
            return

        # 2. Split
        chunks = split_documents(documents)
        if not chunks:
            print("No chunks created. Exiting.")
            return

        # 3. Create Vector Store
        vector_store = create_vector_store(chunks)
        if vector_store:
            print("Pipeline completed successfully.")
        else:
            print("Pipeline failed at vector store creation.")
        
        print("Ingestion pipeline completed")

    except Exception as e:
        print(f"An unexpected error occurred in the pipeline: {e}")

if __name__ == "__main__":
    ingestion()
