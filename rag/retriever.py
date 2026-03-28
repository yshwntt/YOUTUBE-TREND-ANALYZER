# function to create retriever from vectorstore
def get_retriever(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    # k=3 means fetch top 3 most relevant chunks

    return retriever