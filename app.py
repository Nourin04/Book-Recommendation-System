import streamlit as st
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

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
        book_data = pickle.load(f)  # Could be a list of titles or a list of dictionaries
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings = pickle.load(f)

    # Check if book_data is a list of dictionaries or just a list of titles
    if isinstance(book_data, list) and all(isinstance(book, dict) and "title" in book for book in book_data):
        return book_data, np.array(embeddings)  # Correct format
    elif isinstance(book_data, list) and all(isinstance(book, str) for book in book_data):
        # Convert list of strings into dictionaries with placeholder thumbnails
        book_data = [{"title": title, "thumbnail": ""} for title in book_data]
        return book_data, np.array(embeddings)
    else:
        st.error("📌 Error: `book_titles.pkl` is not in the expected format!")
        return [], np.array([])

# Load models and data
faiss_index = load_faiss_index()
book_data, embeddings = load_data()

if not book_data:
    st.error("❌ No book data found! Please check the `book_titles.pkl` file.")
    st.stop()

# Extract book titles for dropdown
book_titles = [book["title"] for book in book_data]

# Streamlit UI
st.title("📚 Book Recommendation System")

# Dropdown for user to select a book
user_selected_book = st.selectbox("Choose a book to get recommendations:", book_titles)

if user_selected_book:
    # Load Sentence Transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Get embedding for selected book
    user_embedding = model.encode(user_selected_book).reshape(1, -1)

    # Search in FAISS index
    D, I = faiss_index.search(user_embedding, 5)

    st.subheader("📌 Recommended Books:")
    recommended_books = []
    
    for i, idx in enumerate(I[0]):
        if idx != -1:
            book_info = book_data[idx]
            recommended_books.append(f"{book_info['title']} (Distance: {D[0][i]:.4f})")

    # Show recommendations in a dropdown
    st.selectbox("🔽 Recommended Books:", recommended_books, index=0)
