import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st

# Extracting text from PDFs
def extract_text_from_pdf(pdf_path):
    try:
        document = fitz.open(pdf_path)
        text = ""
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

pdf_paths = {
    "Alphabet Inc.": "C:/Users/user/Desktop/content_engine/goog-10-k-2023 (1).pdf",
    "Tesla, Inc.": "C:/Users/user/Desktop/content_engine/tsla-20231231-gen.pdf",
    "Uber Technologies, Inc.": "C:/Users/user/Desktop/content_engine/uber-10-k-2023.pdf"
}

pdf_texts = {company: extract_text_from_pdf(path) for company, path in pdf_paths.items()}

# Generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(texts):
    embeddings = {key: model.encode(text) for key, text in texts.items() if text}
    return embeddings

embeddings = generate_embeddings(pdf_texts)

# Storing in vector store
def store_embeddings(embeddings):
    dimension = len(next(iter(embeddings.values())))
    index = faiss.IndexFlatL2(dimension)
    vectors = np.array(list(embeddings.values()))
    index.add(vectors)
    return index

index = store_embeddings(embeddings)

# Query engine (example placeholder)
def query_engine(query, index, embeddings):
    query_vector = model.encode([query])[0]
    distances, indices = index.search(np.array([query_vector]), k=3)
    results = {list(embeddings.keys())[idx]: distances[0][i] for i, idx in enumerate(indices[0])}
    return results

# Streamlit interface
st.title("Content Engine Chatbot")

user_query = st.text_input("Ask a question:")

if user_query:
    try:
        results = query_engine(user_query, index, embeddings)
        for doc, distance in results.items():
            st.write(f"Document: {doc}")
            st.write(f"Relevance: {distance}")
    except Exception as e:
        st.error(f"Error querying engine: {str(e)}")
