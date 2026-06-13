from ingestion.loader import load_documents
from ingestion.splitter import split_documents
from retrieval.vector_store import create_vector_store
from config.settings import settings
from generation.qa_pipeline import answer_question
import sys
import os
import argparse
import glob
import json

def get_data_files_manifest(path="data/raw"):
    manifest = {}
    if not os.path.exists(path):
        return manifest
    search_path = os.path.join(path, "**", "*.pdf")
    pdf_files = glob.glob(search_path, recursive=True)
    for file_path in pdf_files:
        try:
            mtime = os.path.getmtime(file_path)
            size = os.path.getsize(file_path)
            rel_path = os.path.relpath(file_path, path)
            manifest[rel_path] = {"mtime": mtime, "size": size}
        except Exception as e:
            print(f"Error reading file info for {file_path}: {e}")
    return manifest

def write_manifest(manifest):
    try:
        os.makedirs(settings.CHROMA_DB_DIR, exist_ok=True)
        manifest_path = os.path.join(settings.CHROMA_DB_DIR, "ingest_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)
        print("Ingestion manifest updated.")
    except Exception as e:
        print(f"Error writing manifest: {e}")

def read_manifest():
    manifest_path = os.path.join(settings.CHROMA_DB_DIR, "ingest_manifest.json")
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading manifest: {e}")
        return None

def should_reingest():
    if not os.path.exists(settings.CHROMA_DB_DIR):
        return True
    try:
        if len(os.listdir(settings.CHROMA_DB_DIR)) == 0:
            return True
    except Exception:
        return True

    saved_manifest = read_manifest()
    if saved_manifest is None:
        return True

    current_manifest = get_data_files_manifest()
    return current_manifest != saved_manifest

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
            # Save manifest of files ingested
            manifest = get_data_files_manifest()
            write_manifest(manifest)
        else:
            print("Pipeline failed at vector store creation.")
        
        print("Ingestion pipeline completed")

    except Exception as e:
        print(f"An unexpected error occurred in the pipeline: {e}")

def run_query(query_str):
    """
    Submits a single query to the RAG system and prints the response and citations.
    """
    print(f"\nQuery: {query_str}")
    print("Retrieving context and generating response...")
    response, docs = answer_question(query_str)
    
    print("\n--- Response ---")
    print(response)
    print("----------------")
    
    if docs:
        print("\nSources:")
        seen_sources = set()
        for doc in docs:
            src = os.path.basename(doc.metadata.get("source", "Unknown Source"))
            page = doc.metadata.get("page", 0) + 1
            source_str = f"- {src} (Page {page})"
            if source_str not in seen_sources:
                print(source_str)
                seen_sources.add(source_str)
    print()

def interactive_mode():
    """
    Starts an interactive loop in the command line for asking multiple questions.
    """
    print("\n=======================================================")
    print("   Interactive RAG QA Session (Type 'exit' or 'q' to quit) ")
    print("=======================================================")
    
    # Load vector store once to reuse
    from retrieval.vector_store import load_vector_store
    print("Loading vector database...")
    vector_store = load_vector_store()
    if not vector_store:
        print("Error: Could not load the vector database. Have you run ingestion?")
        return
        
    print("System ready. Ask any question!\n")
    while True:
        try:
            query = input("Ask a question > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive mode. Goodbye!")
                break
                
            print("\nRetrieving context and generating response...")
            response, docs = answer_question(query, vector_store=vector_store)
            
            print("\n--- Response ---")
            print(response)
            print("----------------")
            
            if docs:
                print("\nSources:")
                seen_sources = set()
                for doc in docs:
                    src = os.path.basename(doc.metadata.get("source", "Unknown Source"))
                    page = doc.metadata.get("page", 0) + 1
                    source_str = f"- {src} (Page {page})"
                    if source_str not in seen_sources:
                        print(source_str)
                        seen_sources.add(source_str)
            print("\n" + "="*55 + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}\n")

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline: Ingestion and QA Generation")
    parser.add_argument("--ingest", action="store_true", help="Run the document ingestion pipeline")
    parser.add_argument("--query", type=str, help="Submit a single question to the RAG system")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive QA session")
    args = parser.parse_args()

    if args.ingest:
        ingestion()
    elif args.query:
        if should_reingest():
            print("Detected changes in data directory. Running ingestion first...")
            ingestion()
        run_query(args.query)
    elif args.interactive:
        if should_reingest():
            print("Detected changes in data directory. Running ingestion first...")
            ingestion()
        interactive_mode()
    else:
        # Default behavior: run interactive, but ingest first if needed
        if should_reingest():
            print("Detected changes in data/raw or database. Running ingestion pipeline...")
            ingestion()
            print("\nIngestion completed. Starting interactive QA mode...")
        interactive_mode()

if __name__ == "__main__":
    main()
