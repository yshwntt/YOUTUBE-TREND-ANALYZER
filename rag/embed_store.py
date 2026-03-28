from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from rag.config import OPENAI_API_KEY, EMBEDDING_MODEL, VECTORSTORE_DIR

# function to create embeddings model
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# function to build and save chroma db
def create_vectorstore(chunks):
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR)
    )
    # chroma will store vectors on disk in vectorstore/chroma_db

    return vectorstore

# function to load existing chroma db
def load_vectorstore():
    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings
    )

    return vectorstore