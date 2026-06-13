from langchain_chroma import Chroma
from ingestion.embedder import get_embedding_model
from config.settings import settings

class RobustEmbeddings:
    """
    Wrapper for embedding models that ensures one embedding per document.
    Chroma expects embeddings per documnets
    Google embedding model returns a single list of embeddings
    Chroma requires a 1-to-1 match, but the Google model was returning a 1-to-Many or "collapsed" result. The wrapper restores that 1-to-1 relationship.
    """
    def __init__(self, model):
        self.model = model
        
    def embed_documents(self, texts):
        print(f"RobustEmbeddings: Embedding {len(texts)} documents individually...")
        embeddings = []
        for i, text in enumerate(texts):
            try:
                emb = self.model.embed_documents([text])
                if emb and len(emb) == 1:
                    embeddings.append(emb[0])
                else:
                    print(f"Warning: Unexpected embedding result for chunk {i}")
                    embeddings.append([0.0] * 768)
            except Exception as e:
                print(f"Error embedding chunk {i}: {e}")
                embeddings.append([0.0] * 768)
        return embeddings
    
    def embed_query(self, text):
        try:
            return self.model.embed_query(text)
        except Exception as e:
            print(f"Error embedding query: {e}")
            return [0.0] * 768

def create_vector_store(chunks):
    try:
        if not chunks:
            print("Warning: No chunks provided for vector store creation.")
            return None
            
        embedding_model = get_embedding_model()
        if not embedding_model:
            raise ValueError("Could not initialize embedding model.")
            
        robust_embeddings = RobustEmbeddings(embedding_model)

        # Clear existing database directory to avoid keeping old or deleted documents
        import shutil
        import os
        if os.path.exists(settings.CHROMA_DB_DIR):
            print(f"Clearing old vector database at {settings.CHROMA_DB_DIR}...")
            try:
                shutil.rmtree(settings.CHROMA_DB_DIR)
                print("Old vector database cleared.")
            except Exception as e:
                print(f"Warning: Could not clear database directory: {e}")

        print(f"Creating vector store from {len(chunks)} chunks...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=robust_embeddings,
            persist_directory=settings.CHROMA_DB_DIR
        )
        print("Vector store created successfully.")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def load_vector_store():
    try:
        embedding_model = get_embedding_model()
        if not embedding_model:
            raise ValueError("Could not initialize embedding model.")
            
        robust_embeddings = RobustEmbeddings(embedding_model)

        return Chroma(
            persist_directory=settings.CHROMA_DB_DIR,
            embedding_function=robust_embeddings
        )
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None