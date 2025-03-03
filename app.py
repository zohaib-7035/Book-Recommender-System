import pickle
import streamlit as st
import numpy as np

st.header('Book Recommender System Using Machine Learning')

# Load the models and data
model = pickle.load(open('model.pkl', 'rb'))
book_names = pickle.load(open('book_names.pkl', 'rb'))
final_rating = pickle.load(open('final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('book_pivot.pkl', 'rb'))

# Debugging statements
st.write("Columns in final_rating DataFrame:")
st.write(final_rating.columns)

def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion[0]:  # Adjusted to suggestion[0] to get the actual indices
        book_name.append(book_pivot.index[book_id])

    for name in book_name:
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['image_url']  # Use the correct column name
        poster_url.append(url)

    return poster_url

def recommend_book(book_name):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    poster_url = fetch_poster(suggestion)

    for i in range(len(suggestion[0])):  # Adjusted to suggestion[0] to get the actual indices
        books = book_pivot.index[suggestion[0][i]]
        books_list.append(books)

    return books_list, poster_url

selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    book_names
)

if st.button('Show Recommendation'):
    recommended_books, poster_url = recommend_book(selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_books[1])
        st.image(poster_url[1])
    with col2:
        st.text(recommended_books[2])
        st.image(poster_url[2])
    with col3:
        st.text(recommended_books[3])
        st.image(poster_url[3])
    with col4:
        st.text(recommended_books[4])
        st.image(poster_url[4])
    with col5:
        st.text(recommended_books[5])
        st.image(poster_url[5])