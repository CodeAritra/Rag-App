def get_retriever(vector_store):
    try:
        if not vector_store:
            print("Warning: No vector store provided for retriever.")
            return None
            
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        return retriever
    except Exception as e:
        print(f"Error creating retriever: {e}")
        return None