


# **📚 Book Recommendation System**
An AI-powered book recommendation system built using **FAISS, Sentence Transformers, and Streamlit**. This app suggests similar books based on the user's selected title, providing details like authors, categories, ratings, descriptions, and cover images.

---



# **Deployed Link**
https://book-recommendation-system2.streamlit.app/

---




## **🌟 Features**
✅ **Content-Based Recommendations** – Uses sentence embeddings for better book similarity detection.  
✅ **Fast Search with FAISS** – Efficient nearest-neighbor search for instant recommendations.  
✅ **User-Friendly UI with Streamlit** – Simple dropdown to select a book and view recommendations.  
✅ **Book Details Display** – Shows title, authors, categories, ratings, descriptions, and cover images.  
✅ **Dynamic Background** – Sets a background image for better aesthetics.  

---



## **🛠️ Tech Stack**
- **Python** – Core programming language  
- **Streamlit** – For the user interface  
- **FAISS (Facebook AI Similarity Search)** – For fast nearest-neighbor search  
- **Sentence Transformers** – For generating book embeddings  
- **Pickle** – For storing book metadata and embeddings  

---




## **📂 Folder Structure**
```
📂 book-recommendation-system
│── 📜 app.py                  # Main Streamlit app
│── 📜 book_recommender.faiss   # FAISS index for fast retrieval
│── 📜 book_titles_thumbnail (1).pkl # Book metadata (title, authors, categories, etc.)
│── 📜 embeddings.pkl           # Precomputed book embeddings
│── 📜 requirements.txt         # Dependencies
│── 📜 README.md                # Documentation
```

---



## **📥 Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/book-recommendation-system.git
cd book-recommendation-system
```

### **2️⃣ Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Run the Application**
```bash
streamlit run app.py
```

---



## **📊 How It Works**
1. **Book Embeddings**  
   - The book dataset is **preprocessed** and converted into **sentence embeddings** using **Sentence Transformers**.
   - These embeddings are stored in `embeddings.pkl`.

2. **FAISS for Fast Search**  
   - FAISS indexes the embeddings to enable **fast book retrieval**.
   - When a book is selected, its embedding is **compared with the FAISS index** to find the top 5 most similar books.

3. **Book Details Display**  
   - The system retrieves **title, authors, categories, ratings, descriptions, and cover images** from `book_titles_thumbnail.pkl`.
   - If a book lacks a **thumbnail**, a warning is displayed instead of breaking the app.

4. **User Interface**  
   - Built with **Streamlit**, allowing users to select a book and instantly get recommendations.
   - A **background image** enhances the UI aesthetics.

---



## **📷 Screenshots**
### **Main Interface**
![Screenshot (63)](https://github.com/user-attachments/assets/bae90b31-7f23-4193-b06c-6926166c2da2)

### **Book Recommendations**
![Screenshot (64)](https://github.com/user-attachments/assets/5d4507c3-4f87-4e04-aa3a-16dd4b47d7ba)


---



## **🛠️ Troubleshooting**
**Error:** `No module named faiss`  
✅ Solution: Install FAISS manually  
```bash
pip install faiss-cpu
```

**Error:** `No thumbnail available`  
✅ Solution: Ensure `book_titles_thumbnail.pkl` contains valid `thumbnail` URLs. Otherwise, a placeholder will be shown.

**Error:** `FileNotFoundError: embeddings.pkl not found`  
✅ Solution: Ensure the dataset is preprocessed, and embeddings are generated.

---



## **📌 Future Improvements**
🚀 **Hybrid Recommendations** – Combine content-based filtering with collaborative filtering.  
🚀 **Better UI/UX** – Improve styling and responsiveness.  
🚀 **User Login System** – Allow users to save preferences.  
🚀 **More Book Data** – Enhance dataset with more book details and genres.  

---



## **📜 License**
This project is open-source and available under the **MIT License**.

---



## **🤝 Contributing**
Want to contribute?  
1. **Fork the repository**  
2. **Create a new branch:** `git checkout -b feature-branch`  
3. **Commit changes:** `git commit -m "Add new feature"`  
4. **Push to GitHub:** `git push origin feature-branch`  
5. **Create a Pull Request**  

---



## **📩 Contact**
💡 **Maintainer:** Your Name  
🔗 **GitHub:** [Your GitHub Profile](https://github.com/Nourin04)  
✉️ **Email:** nourinnn1823@gmail.com  

---

