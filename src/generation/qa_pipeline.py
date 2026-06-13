from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retrieval.vector_store import load_vector_store
from retrieval.retriever import get_retriever
from config.settings import settings
import os

def get_llm():
    """
    Initializes and returns the ChatGoogleGenerativeAI model.
    """
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not configured in settings. Check your .env file.")
    
    return ChatGoogleGenerativeAI(
        model=settings.MODEL_NAME,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.2
    )

def format_docs(docs):
    """
    Formats list of Document objects into a single context string with clear source metadata.
    """
    formatted = []
    for i, doc in enumerate(docs):
        source = os.path.basename(doc.metadata.get("source", "Unknown Source"))
        page = doc.metadata.get("page", 0) + 1  # LangChain page numbers are 0-indexed, make it 1-indexed for display
        formatted.append(f"Document [{i+1}] (Source: {source}, Page: {page}):\n{doc.page_content}")
    return "\n\n".join(formatted)

def answer_question(query: str, vector_store=None):
    """
    Given a user query, retrieves relevant contexts from Chroma DB and generates a response.
    Returns:
        tuple: (response_text, retrieved_documents)
    """
    try:
        if not vector_store:
            vector_store = load_vector_store()
            if not vector_store:
                raise ValueError("Failed to load Chroma vector store. Ensure you have run ingestion first.")
        
        retriever = get_retriever(vector_store)
        if not retriever:
            raise ValueError("Failed to initialize retriever from vector store.")
        
        # 1. Retrieve matching documents
        retrieved_docs = retriever.invoke(query)
        if not retrieved_docs:
            return "No relevant documents were found in the database to answer your question.", []
        
        # 2. Format documents into a context block
        context = format_docs(retrieved_docs)
        
        # 3. Create RAG prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a helpful and precise assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the user's question. "
                "If you do not know the answer or if the context does not provide sufficient info, "
                "state that you cannot find the answer in the provided documents. Do not make up facts.\n\n"
                "Retrieved Context:\n{context}"
            )),
            ("human", "{question}")
        ])
        
        # 4. Initialize LLM
        llm = get_llm()
        
        # 5. Build LCEL Chain
        chain = prompt | llm | StrOutputParser()
        
        # 6. Execute chain
        response_text = chain.invoke({"context": context, "question": query})
        
        return response_text, retrieved_docs
        
    except Exception as e:
        print(f"Error in RAG generation pipeline: {e}")
        return f"An error occurred while generating the answer: {e}", []
