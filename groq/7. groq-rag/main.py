import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

# Load environment variables from the root .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "../../", ".env"))

def run_rag_demo():
    # 0. Configuration
    index_path = "faiss_index"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 1. Check if index exists locally
    if os.path.exists(index_path):
        print(f"✓ Loading existing FAISS index from {index_path}...")
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("! FAISS index not found. Creating new one...")
        # 1a. Prepare dummy data
        text_data = """
        Groq is a technology company that builds the LPU (Language Processing Unit), 
        which is an ultra-fast AI inference engine designed to deliver real-time 
        performance for Large Language Models (LLMs).
        
        FAISS (Facebook AI Similarity Search) is a library for efficient similarity 
        search and clustering of dense vectors. It is commonly used as a vector 
        database for RAG systems.
        
        RAG (Retrieval-Augmented Generation) is a technique that gives LLMs access 
        to external data to improve their responses and reduce hallucinations.
        
        The user asking for this demo is named Pranesh, a developer working on 
        Groq and LangChain integrations.
        """

        # 1b. Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        docs = text_splitter.create_documents([text_data])
        print(f"✓ Created {len(docs)} chunks from text.")

        # 1c. Create and save FAISS vector store
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(index_path)
        print(f"✓ FAISS vector store created and saved to {index_path}.")

    # 5. Initialize ChatGroq LLM
    llm = ChatGroq(
        model_name=os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    # 6. Setup RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # 7. Ask a question
    query = "Who is the user and what is he working on?"
    print(f"\nQuestion: {query}")
    
    response = qa_chain.invoke(query)
    print(f"\nAnswer: {response['result']}")

    query2 = "What is the LPU and who builds it?"
    print(f"\nQuestion: {query2}")
    
    response2 = qa_chain.invoke(query2)
    print(f"\nAnswer: {response2['result']}")

if __name__ == "__main__":
    run_rag_demo()
