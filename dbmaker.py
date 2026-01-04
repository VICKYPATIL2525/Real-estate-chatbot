import json
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ====== FILE PATHS ======
JSON_FILE_PATH = "./mahendra_rechunked2.json"
PERSIST_DIRECTORY = "./chroma_db"


def create_vector_db(json_path: str, persist_directory: str):
    """Create and save a Chroma vector DB from JSON with text + url metadata."""

    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        text = item["text"].strip()
        urls = item["urls"]  # list → contains 1 URL

        # Convert list to string (Chroma requirement)
        url_string = ", ".join(urls)

        documents.append(
            Document(
                page_content=text,
                metadata={"urls": url_string}
            )
        )

    print(f"Processed {len(documents)} JSON chunks")

    # Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create Chroma DB
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(f"\n✅ Vector DB created at: {os.path.abspath(persist_directory)}")
    print("👉 You can now load this DB anytime for querying.\n")


if __name__ == "__main__":
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    create_vector_db(JSON_FILE_PATH, PERSIST_DIRECTORY)
