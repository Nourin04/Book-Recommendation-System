import streamlit as st
import faiss
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load FAISS index
INDEX_FILE = "book_recommender.faiss"
TITLE_FILE = "book_titles_thumbnail.pkl"  # Updated to new file with thumbnails
EMBEDDINGS_FILE = "embeddings.pkl"

@st.cache_resource
def load_faiss_index():
    index = faiss.read_index(INDEX_FILE)
    return index

@st.cache_data
def load_data():
    with open(TITLE_FILE, "rb") as f:
        book_data = pickle.load(f)  # Load title + thumbnail data
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings = pickle.load(f)
    return book_data, np.array(embeddings)

# Load models and data
faiss_index = load_faiss_index()
book_data, embeddings = load_data()

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.title("📚 Book Recommendation System")

# Get user input as a dropdown (autocomplete)
book_titles = [book["title"] for book in book_data]
user_input = st.selectbox("Select a book to get recommendations:", book_titles)

if user_input:
    # Get embedding for user input
    user_embedding = model.encode(user_input).reshape(1, -1)

    # Search in FAISS index
    D, I = faiss_index.search(user_embedding, 5)

    st.subheader("Recommended Books:")
    for i, idx in enumerate(I[0]):
        if idx != -1:
            title = book_data[idx]["title"]
            thumbnail = book_data[idx]["thumbnail"]
            st.write(f"**{i+1}. {title}** (Distance: {D[0][i]:.4f})")
            st.image(thumbnail, width=100)  # Display thumbnail
        else:
            st.write("No recommendations found.")
