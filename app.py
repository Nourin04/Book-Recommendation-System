import streamlit as st
import faiss
import numpy as np
import pickle

# Load FAISS index
INDEX_FILE = "book_recommender.faiss"
TITLE_FILE = "book_titles.pkl"
EMBEDDINGS_FILE = "embeddings.pkl"

@st.cache_resource
def load_faiss_index():
    index = faiss.read_index(INDEX_FILE)
    return index

@st.cache_data
def load_data():
    with open(TITLE_FILE, "rb") as f:
        book_titles = pickle.load(f)
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings = pickle.load(f)
    return book_titles, np.array(embeddings)

# Load models and data
faiss_index = load_faiss_index()
book_titles, embeddings = load_data()

# Streamlit UI
st.title("📚 Book Recommendation System")

user_input = st.text_input("Enter a book title to get recommendations:")

if user_input:
    # Find the closest match
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer

    # Load Sentence Transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Get embedding for user input
    user_embedding = model.encode(user_input).reshape(1, -1)

    # Search in FAISS index
    D, I = faiss_index.search(user_embedding, 5)

    st.subheader("Recommended Books:")
    for i, idx in enumerate(I[0]):
        if idx != -1:
            st.write(f"{i+1}. {book_titles[idx]} (Distance: {D[0][i]:.4f})")
        else:
            st.write("No recommendations found.")

