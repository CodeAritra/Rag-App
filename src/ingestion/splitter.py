from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    try:
        if not documents:
            print("Warning: No documents provided for splitting.")
            return []
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)
        print(f"Successfully split into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return []