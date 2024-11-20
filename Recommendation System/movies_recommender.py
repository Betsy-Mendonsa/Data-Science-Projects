import streamlit as st
import pickle
import pandas as pd
import difflib as dlb

# Load the saved model, vectorizer and the dataset
with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    loaded_tfidf_vect = pickle.load(vec_file)

with open('similarity_scores.pkl', 'rb') as sim_file:
    loaded_similarity_scores = pickle.load(sim_file)

df = pd.read_csv('movies.csv')


st.title("🎬 Movie Recommendation System")
st.write("Get recommendations for similar movies based on your favorite movie")

movie_name = st.text_input("Enter a movie name:", "")

if st.button("Get Recommendations"):
    if movie_name.strip() == "":
        st.error("Please enter a valid movie name!")
    else:
        # Find the closest match for the entered movie name
        titles = df['title'].tolist()
        get_close_match = dlb.get_close_matches(movie_name, titles)

        if len(get_close_match) == 0:
            st.error("No matching movie found. Please try again!")
        else:
            closest_match = get_close_match[0]
            index_of_movie = df[df.title == closest_match].index[0]
            sim_score = list(enumerate(loaded_similarity_scores[index_of_movie]))
            similar_movies = sorted(sim_score, key=lambda x: x[1], reverse=True)

            # Collect details for the top 10 recommendations
            recommendations = []
            i = 1
            for movie in similar_movies:
                if i > 10:
                    break
                Index = movie[0]
                movie_details = df.iloc[Index][['title', 'genres', 'vote_average']]
                recommendations.append({
            
                    "Title": movie_details['title'],
                    "Rating": f"{movie_details['vote_average']:.1f}",
                    "Genre": movie_details['genres']
                })
                i += 1

            # Display recommendations in a table
            if recommendations:
                st.subheader(f"If you liked '{closest_match}', You might also love these movies:")
                st.table(pd.DataFrame(recommendations))
            else:
                st.error("No recommendations found!")


