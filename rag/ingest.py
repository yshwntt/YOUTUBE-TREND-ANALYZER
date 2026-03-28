from pathlib import Path
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

# function to load all supported files from data/raw
def load_documents(data_dir):
    documents = []

    # loop through every file in raw data folder
    for file_path in Path(data_dir).glob("*"):
        suffix = file_path.suffix.lower()

        # load pdf files
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())

        # load txt files
        elif suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())

        # load csv files
        elif suffix == ".csv":
            df = pd.read_csv(file_path)

            # convert each row into one document
            for i, row in df.iterrows():
                row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
                documents.append(
                    Document(
                        page_content=row_text,
                        metadata={"source": str(file_path), "row": i}
                    )
                )

    return documents