import streamlit as st
import faiss
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Constants
INDEX_FILE = "book_recommender.faiss"
TITLE_FILE = "book_titles.pkl"
EMBEDDINGS_FILE = "embeddings.pkl"
THUMBNAIL_FILE = "books.csv"  # Ensure this CSV has 'title' and 'thumbnail' columns

# Load FAISS index
@st.cache_resource
def load_faiss_index():
    index = faiss.read_index(INDEX_FILE)
    return index

# Load book titles and embeddings
@st.cache_data
def load_data():
    with open(TITLE_FILE, "rb") as f:
        book_titles = pickle.load(f)
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings = pickle.load(f)
    return book_titles, np.array(embeddings)

# Load book thumbnails from CSV
@st.cache_data
def load_thumbnails():
    books_df = pd.read_csv(THUMBNAIL_FILE)
    return dict(zip(books_df["title"], books_df["thumbnail"]))  # Ensure correct column names

# Load all models and data
faiss_index = load_faiss_index()
book_titles, embeddings = load_data()
book_thumbnails = load_thumbnails()

# Streamlit UI
st.title("📚 Book Recommendation System")

user_input = st.text_input("Enter a book title to get recommendations:")

if user_input:
    # Load Sentence Transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Get embedding for user input
    user_embedding = model.encode(user_input).reshape(1, -1)

    # Search in FAISS index
    D, I = faiss_index.search(user_embedding, 5)

    st.subheader("📖 Recommended Books:")
    for i, idx in enumerate(I[0]):
        if idx != -1:
            book_title = book_titles[idx]
            thumbnail_url = book_thumbnails.get(book_title, "https://via.placeholder.com/150")  # Default image
            
            # Display book title and thumbnail
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(thumbnail_url, width=100)
            with col2:
                st.write(f"**{book_title}**")
                st.write(f"🔹 Distance: {D[0][i]:.4f}")
        else:
            st.write("No recommendations found.")
