from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import os

def load_documents(path="data/raw"):
    try:
        if not os.path.exists(path):
            print(f"Warning: Directory {path} does not exist.")
            return []
            
        loader = DirectoryLoader(
            path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )

        documents = loader.load()
        print(f"Successfully loaded {len(documents)} documents from {path}")
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []