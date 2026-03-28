from langchain_text_splitters import RecursiveCharacterTextSplitter

# function to split documents into smaller chunks
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    # splitting loaded documents into overlapping chunks
    chunks = splitter.split_documents(documents)
    return chunks