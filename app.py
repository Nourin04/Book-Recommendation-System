import streamlit as st
import faiss
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Function to set background image
def set_background(image_url):
    """Set a full-screen background image using CSS."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image (replace with your image URL)
set_background("https://images.pexels.com/photos/762686/pexels-photo-762686.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1")

# File paths
INDEX_FILE = "book_recommender.faiss"
TITLE_FILE = "book_titles_thumbnail.pkl"  # Ensure correct filename
EMBEDDINGS_FILE = "embeddings.pkl"

@st.cache_resource
def load_faiss_index():
    """Load FAISS index."""
    return faiss.read_index(INDEX_FILE)

@st.cache_data
def load_data():
    """Load book data and embeddings."""
    with open(TITLE_FILE, "rb") as f:
        book_data = pickle.load(f)
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings = pickle.load(f)
    return book_data, np.array(embeddings)

# Load FAISS index and book data
faiss_index = load_faiss_index()
book_data, embeddings = load_data()

# Load Sentence Transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Streamlit UI
st.title("📚 Book Recommendation System")

# Book title dropdown for selection
book_titles = [book["title"] for book in book_data]
selected_book = st.selectbox("🔍 Select a book to find similar recommendations:", book_titles)

if selected_book:
    # Get embedding for selected book
    book_idx = book_titles.index(selected_book)
    book_embedding = embeddings[book_idx].reshape(1, -1)

    # Search in FAISS index
    D, I = faiss_index.search(book_embedding, 5)

    st.subheader("📖 Recommended Books:")
    for i, idx in enumerate(I[0]):
        if idx != -1:  # Ensure valid index
            recommended_book = book_data[idx]

            # Get book details with error handling
            title = recommended_book.get("title", "Unknown Title")
            authors = ", ".join(recommended_book.get("authors", ["Unknown"]))
            categories = ", ".join(recommended_book.get("categories", ["N/A"]))
            average_rating = recommended_book.get("average_rating", "N/A")
            ratings_count = recommended_book.get("ratings_count", "N/A")
            description = recommended_book.get("description", "No description available.")[:300] + "..."
            thumbnail_url = recommended_book.get("thumbnail", "")

            # Display book details
            st.markdown(f"### {title}")

            if thumbnail_url:
                st.image(thumbnail_url, width=100)
            else:
                st.warning("🚨 No thumbnail available for this book.")

            st.write(f"**📚 Author(s):** {authors}")
            st.write(f"**🏷️ Category:** {categories}")
            st.write(f"**⭐ Average Rating:** {average_rating} ({ratings_count} ratings)")
            st.write(f"**📄 Description:** {description}")
            st.write("---")
