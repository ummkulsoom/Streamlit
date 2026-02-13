import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load precomputed document embeddings
embeddings = np.load("embeddings.npy")

# Load documents (one document per line)
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f.readlines() if line.strip()]

def retrieve_top_k(query_embedding, embeddings, documents, k=5):
    sims = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_k_idx = sims.argsort()[-k:][::-1]
    return [(documents[i], float(sims[i])) for i in top_k_idx]

st.title("Information Retrieval using Document Embeddings")

query = st.text_input("Enter your query:")
k = st.slider("Top K results", 1, 10, 5)

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        # Make a query embedding based on keyword matches in documents
        query_embedding = np.zeros(embeddings.shape[1], dtype=np.float32)
        query_words = query.lower().split()

        for i, doc in enumerate(documents):
            doc_lower = doc.lower()
            for w in query_words:
                if w in doc_lower:
                    query_embedding += embeddings[i]

        # If no keywords matched any document, fallback to random
        if np.all(query_embedding == 0):
            query_embedding = np.random.rand(embeddings.shape[1]).astype(np.float32)

        results = retrieve_top_k(query_embedding, embeddings, documents, k=k)

        st.success("Search completed")
        st.write(f"### Top {k} Relevant Documents:")
        for rank, (doc, score) in enumerate(results, start=1):
            st.write(f"{rank}. {doc} (Score: {score:.4f})")
