import streamlit as st
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# File paths
INDEX_FILE = "book_recommender.faiss"
TITLE_FILE = "book_titles.pkl"
EMBEDDINGS_FILE = "embeddings.pkl"

@st.cache_resource
def load_faiss_index():
    return faiss.read_index(INDEX_FILE)

@st.cache_data
def load_data():
    with open(TITLE_FILE, "rb") as f:
        book_data = pickle.load(f)
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings = pickle.load(f)

    # Ensure correct format (list of dicts with 'title' & 'thumbnail')
    if isinstance(book_data, list) and all(isinstance(book, dict) and "title" in book and "thumbnail" in book for book in book_data):
        return book_data, np.array(embeddings)
    else:
        st.error("❌ `book_titles.pkl` format is incorrect! It must contain 'title' and 'thumbnail' keys.")
        return [], np.array([])

# Load FAISS index and book data
faiss_index = load_faiss_index()
book_data, embeddings = load_data()

if not book_data:
    st.error("📌 No book data found! Please check `book_titles.pkl`")
    st.stop()

# Extract book titles for dropdown
book_titles = [book["title"] for book in book_data]

# Streamlit UI
st.title("📚 Book Recommendation System")

# Dropdown to select a book
user_selected_book = st.selectbox("Choose a book to get recommendations:", book_titles)

if user_selected_book:
    # Load Sentence Transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Get embedding for selected book
    user_embedding = model.encode(user_selected_book).reshape(1, -1)

    # Search in FAISS index
    D, I = faiss_index.search(user_embedding, 5)

    st.subheader("📌 Recommended Books:")

    # Layout: Show books in a row (5 columns)
    cols = st.columns(5)  # Create 5 columns

    for i, idx in enumerate(I[0]):
        if idx != -1:
            book_info = book_data[idx]
            with cols[i]:  # Assign each book to a column
                st.image(book_info["thumbnail"], width=120)
                st.write(f"**{book_info['title']}**")  # Bold title
                st.caption(f"🔹 Distance: {D[0][i]:.4f}")  # Show distance score
