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
set_background("https://images.pexels.com/photos/1831744/pexels-photo-1831744.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1")

# File paths
INDEX_FILE = "book_recommender.faiss"
TITLE_FILE = "book_titles_thumbnail (1).pkl"
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
        if idx != -1:
            recommended_book = book_data[idx]
            
            # Display book details
            st.markdown(f"### {recommended_book['title']}")
            st.image(recommended_book['thumbnail'], width=100)
            st.write(f"**📚 Author(s):** {', '.join(recommended_book['authors']) if recommended_book['authors'] else 'Unknown'}")
            st.write(f"**🏷️ Category:** {', '.join(recommended_book['categories']) if recommended_book['categories'] else 'N/A'}")
            st.write(f"**⭐ Average Rating:** {recommended_book['average_rating']} ({recommended_book['ratings_count']} ratings)")
            st.write(f"**📄 Description:** {recommended_book['description'][:300]}...")  # Show only first 300 chars
            st.write("---")
