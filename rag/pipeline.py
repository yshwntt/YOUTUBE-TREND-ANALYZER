from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rag.retriever import get_retriever

# function to load local llm
def get_llm():
    model_name = "google/flan-t5-small"
    # small free local model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model

# function to answer question using retrieved context
def answer_question(vectorstore, question):
    # create retriever
    retriever = get_retriever(vectorstore)

    # get relevant docs
    docs = retriever.invoke(question)

    # combine retrieved chunks into one context
    context = "\n\n".join([doc.page_content for doc in docs])

    # prompt
    prompt = f"""
Answer the question only using the context below.
If the answer is not in the context, say:
I could not find that in the provided documents.

Context:
{context}

Question:
{question}

Answer:
"""

    # load tokenizer + model
    tokenizer, model = get_llm()

    # tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    # generate answer
    outputs = model.generate(**inputs, max_new_tokens=150)

    # decode answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer, docs