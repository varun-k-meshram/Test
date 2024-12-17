import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommendationSystem:
    def __init__(self):
        # Initialize the movie dataset
        self.movies = self._create_movie_dataset()
        
        # Prepare feature matrix for recommendations
        self._prepare_recommendation_features()
    
    def _create_movie_dataset(self):
        """
        Create a dataset of 20 movies with various attributes
        """
        movies_data = [
           {"title": "The Shawshank Redemption", "genre": "Drama", "year": 1994, "rating": 9.3, 
             "description": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
             "poster": "https://m.media-amazon.com/images/M/MV5BNDE3ODQxNDk5NF5BMl5BanBnXkFtZTcwNjk3NzM3OA@@._V1_FMjpg_UX1000_.jpg"},
            {"title": "The Godfather", "genre": "Crime Drama", "year": 1972, "rating": 9.2, 
             "description": "An organized crime dynasty's aging patriarch transfers control of his clandestine empire to his reluctant son.",
             "poster": "https://m.media-amazon.com/images/M/MV5BM2MyNjYxNmUtYTAwNi00MTYxLWJmNWYtYzZlODY3ZDFhODAxXkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_FMjpg_UX1000_.jpg"},
            {"title": "The Dark Knight", "genre": "Action Superhero", "year": 2008, "rating": 9.0, 
             "description": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
             "poster": "https://m.media-amazon.com/images/M/MV5BMTMxNTMwODM0NF5BMl5BanBnXkFtZTcwODAyMTk2Mw@@._V1_FMjpg_UX1000_.jpg"},
            {"title": "Pulp Fiction", "genre": "Crime Drama", "year": 1994, "rating": 8.9, 
             "description": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
             "poster": "https://m.media-amazon.com/images/M/MV5BNGNhMDIzZTUtNWEzNy00M2FmLTg5NGQtNmY1NWVmMmU4MjhiXkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_FMjpg_UX1000_.jpg"},
            {"title": "Forrest Gump", "genre": "Drama Romance", "year": 1994, "rating": 8.8, 
             "description": "The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an amazing life.",
             "poster": "https://m.media-amazon.com/images/M/MV5BNWIwODRlZTUtY2U3ZS00Yzg1LWJhNzYtMmZiYmEyNmY1ZmI3XkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_FMjpg_UX1000_.jpg"},
            {"title": "Inception", "genre": "Sci-Fi Action", "year": 2010, "rating": 8.8, 
             "description": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.",
             "poster": "https://m.media-amazon.com/images/M/MV5BMjAxMzY3NjcxNF5BMl5BanBnXkFtZTcwNTI5OTM0Mw@@._V1_FMjpg_UX1000_.jpg"},
            {"title": "The Matrix", "genre": "Sci-Fi Action", "year": 1999, "rating": 8.7, 
             "description": "A computer programmer discovers that reality as he knows it is a simulation created by machines, and joins a rebellion to break free.",
             "poster": "https://m.media-amazon.com/images/M/MV5BNzQzOTk3OTAtNDQ0Zi00ZTVkLWI0MTEtMDllZjNkMzNiNDRhXkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_FMjpg_UX1000_.jpg"},
            {"title": "Goodfellas", "genre": "Crime Drama", "year": 1990, "rating": 8.7, 
             "description": "The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill and his mob partners Jimmy Conway and Tommy DeVito.",
             "poster": "https://m.media-amazon.com/images/M/MV5BY2NkZjEzMDgtN2RjYy00YzM1LWI4ZmQtMjIwYjFjNmI3ZDEzXkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_FMjpg_UX1000_.jpg"},
            {"title": "The Silence of the Lambs", "genre": "Psychological Thriller", "year": 1991, "rating": 8.6, 
             "description": "A young F.B.I. cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer.",
             "poster": "https://m.media-amazon.com/images/M/MV5BNjNhZTk0ZmEtNzhlZS00ZmRmLWJhZjAtMzQ4NzA1NmVmZmJhXkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_FMjpg_UX1000_.jpg"},
            {"title": "Star Wars: Episode IV - A New Hope", "genre": "Sci-Fi Adventure", "year": 1977, "rating": 8.6, 
             "description": "Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, a Wookiee and two droids to save the galaxy from the Empire's world-destroying battle station.",
             "poster": "https://m.media-amazon.com/images/M/MV5BNzVlY2MwMjktM2E4OS00Y2Y3LWE3ZjctOWY2MWJlOTRhOGNiXkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_FMjpg_UX1000_.jpg"},
            {"title": "Fight Club", "genre": "Drama", "year": 1999, "rating": 8.8, 
             "description": "An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much, much more.",
             "poster": "https://m.media-amazon.com/images/M/MV5BMmEzNTM5OTItMTdmNy00YmY2LThjOGYtNzViMmUwYmUwMmY4XkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_FMjpg_UX1000_.jpg"},
            {"title": "Interstellar", "genre": "Sci-Fi Adventure", "year": 2014, "rating": 8.7, 
             "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
             "poster": "https://m.media-amazon.com/images/M/MV5BZjdkOTU3MDAtM2UwMi00YzIxLWFmNDktMWY2NmFmNzVmNmRkXkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_FMjpg_UX1000_.jpg"},
            {"title": "The Lord of the Rings: The Fellowship of the Ring", "genre": "Fantasy Adventure", "year": 2001, "rating": 8.8, 
             "description": "A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron.",
             "poster": "https://m.media-amazon.com/images/M/MV5BN2EyZjM3NzUtNWUzMi00MTgxLWI0NTctMzY4M2VlOTdjZWRiXkEyXkFqcGdeQXVyNDUzOTQ5MjY@._V1_FMjpg_UX1000_.jpg"},
            {"title": "Gladiator", "genre": "Historical Action", "year": 2000, "rating": 8.5, 
             "description": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery.",
             "poster": "https://m.media-amazon.com/images/M/MV5BMDliMmNhNDEtODUyOS00MjNlLWI3ZTQtM2Q3ZmNhMGNmOGVmXkEyXkFqcGdeQXVyNTY3MTYzMDU@._V1_FMjpg_UX1000_.jpg"},
            {"title": "The Departed", "genre": "Crime Thriller", "year": 2006, "rating": 8.5, 
             "description": "An undercover cop and a mole in the police attempt to identify each other while infiltrating an Irish gang in South Boston.",
             "poster": "https://m.media-amazon.com/images/M/MV5BMTI1MTY2OTIxNV5BMl5BanBnXkFtZTYwNzQ4Mzc2._V1_FMjpg_UX1000_.jpg"},
            {"title": "The Green Mile", "genre": "Drama", "year": 1999, "rating": 8.6, 
             "description": "The lives of guards on Death Row are affected by one of their charges: a black man accused of child murder and rape, yet who has a mysterious gift.",
             "poster": "https://m.media-amazon.com/images/M/MV5BMTUwNjU5NTkyMF5BMl5BanBnXkFtZTcwNTc3MDQ2Ng@@._V1_FMjpg_UX1000_.jpg"},
            {"title": "Saving Private Ryan", "genre": "War Drama", "year": 1998, "rating": 8.6, 
             "description": "Following the Normandy Landings, a group of U.S. soldiers go behind enemy lines to retrieve a paratrooper whose brothers have been killed in action.",
             "poster": "https://m.media-amazon.com/images/M/MV5BZjhkMWQ4MzMtZmRmZC00M2UxLTgxOTQtNmNiNDk0Y2NhYzNkXkEyXkFqcGdeQXVyNDYyMDk5MTU@._V1_FMjpg_UX1000_.jpg"},
            {"title": "Spirited Away", "genre": "Animated Fantasy", "year": 2001, "rating": 8.6, 
             "description": "During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits, and where humans are changed into beasts.",
             "poster": "https://m.media-amazon.com/images/M/MV5BMjlmZmI5MDctNDE2YS00YWE0LWE5ZWItZDBhYWQ0NTcxYWFmXkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_FMjpg_UX1000_.jpg"},
            {"title": "The Avengers", "genre": "Superhero Action", "year": 2012, "rating": 8.0, 
             "description": "Earth's mightiest heroes must come together and learn to fight as a team to stop the mischievous Loki and his alien army from enslaving humanity.",
             "poster": "https://m.media-amazon.com/images/M/MV5BNDYxNjQyMjAtNTdiOS00NGYwLWFmNTAtNDQ0ZmQxYWNmNjkyXkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_FMjpg_UX1000_.jpg"},
            {"title": "Avatar", "genre": "Sci-Fi Adventure", "year": 2009, "rating": 7.8, 
             "description": "A paraplegic marine dispatched to the moon Pandora on a unique mission becomes torn between following his mission and protecting the world he feels is his home.",
             "poster": "https://m.media-amazon.com/images/M/MV5BMTYwOTEwNjAzMl5BMl5BanBnXkFtZTcwODc5MTUwMw@@._V1_FMjpg_UX1000_.jpg"}
        ]
        
        return pd.DataFrame(movies_data)
    
    def _prepare_recommendation_features(self):
        """
        Prepare feature matrix for content-based recommendations
        """
        # Combine relevant features for recommendation
        self.movies['features'] = (
            self.movies['title'] + ' ' + 
            self.movies['genre'] + ' ' + 
            self.movies['description']
        )
        
        # Create count matrix from features
        self.count_vectorizer = CountVectorizer(stop_words='english')
        self.count_matrix = self.count_vectorizer.fit_transform(self.movies['features'])
        
        # Compute cosine similarity
        self.cosine_sim = cosine_similarity(self.count_matrix)
    
    def get_recommendations(self, movie_title, top_n=5):
        """
        Get movie recommendations based on similarity
        """
        try:
            movie_index = self.movies[self.movies['title'] == movie_title].index[0]
        except IndexError:
            return []
        
        # Get similarity scores for the movie
        sim_scores = list(enumerate(self.cosine_sim[movie_index]))
        
        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar movies (excluding the input movie itself)
        top_similar_indices = [i[0] for i in sim_scores[1:top_n+1]]
        
        return self.movies.iloc[top_similar_indices]
    
    def recommend_by_genre(self, genre, top_n=5):
        """
        Recommend movies by genre
        """
        genre_movies = self.movies[self.movies['genre'].str.contains(genre, case=False)]
        return genre_movies.nlargest(top_n, 'rating')

def main():
    # Set page configuration
    st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")
    
    # Create recommender instance
    recommender = MovieRecommendationSystem()
    
    # Title and description
    st.title("🎬 Movie Recommendation Portal")
    st.write("Discover your next favorite movie!")
    
    # Sidebar for navigation
    st.sidebar.header("Recommendation Options")
    
    # Recommendation Method Selection
    rec_method = st.sidebar.selectbox(
        "Choose Recommendation Method",
        ["Similar Movies", "By Genre", "Top Rated"]
    )
    
    # Recommendation Display Area
    if rec_method == "Similar Movies":
        # Movie selection for similar recommendations
        selected_movie = st.selectbox(
            "Select a Movie", 
            recommender.movies['title'].tolist()
        )
        
        if st.button("Get Similar Movie Recommendations"):
            # Get recommendations
            similar_movies = recommender.get_recommendations(selected_movie)
            
            # Display recommendations
            st.subheader(f"Movies Similar to {selected_movie}")
            
            # Create columns for movie display
            cols = st.columns(5)
            
            for i, (_, movie) in enumerate(similar_movies.iterrows()):
                with cols[i % 5]:
                    st.image(movie['poster'], use_column_width=True)
                    st.write(movie['title'])
                    st.caption(f"Genre: {movie['genre']}")
                    st.caption(f"Rating: {movie['rating']}")
    
    elif rec_method == "By Genre":
        # Genre selection
        selected_genre = st.sidebar.selectbox(
            "Select Genre", 
            ["Sci-Fi", "Drama", "Action", "Crime", "Adventure", "Superhero"]
        )
        
        if st.button("Find Top Movies in Genre"):
            # Get genre recommendations
            genre_movies = recommender.recommend_by_genre(selected_genre)
            
            # Display recommendations
            st.subheader(f"Top {selected_genre} Movies")
            
            # Create columns for movie display
            cols = st.columns(5)
            
            for i, (_, movie) in enumerate(genre_movies.iterrows()):
                with cols[i % 5]:
                    st.image(movie['poster'], use_column_width=True)
                    st.write(movie['title'])
                    st.caption(f"Genre: {movie['genre']}")
                    st.caption(f"Rating: {movie['rating']}")
    
    else:
        # Top Rated Movies
        st.subheader("Top Rated Movies")
        top_movies = recommender.movies.nlargest(10, 'rating')
        
        # Create columns for movie display
        cols = st.columns(5)
        
        for i, (_, movie) in enumerate(top_movies.iterrows()):
            with cols[i % 5]:
                st.image(movie['poster'], use_column_width=True)
                st.write(movie['title'])
                st.caption(f"Genre: {movie['genre']}")
                st.caption(f"Rating: {movie['rating']}")

if __name__ == "__main__":
    main()