import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv('combined_df.csv')

genre_data = data['combined_genres'].tolist()

tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,  # You can adjust this based on your dataset size and requirements
    stop_words='english',  # Remove common English stop words
)

tfidf_matrix = tfidf_vectorizer.fit_transform(genre_data)

# Create a DataFrame for TF-IDF features
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Concatenate the TF-IDF DataFrame with the original data
result_df = pd.concat([data, tfidf_df], axis=1)

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a reverse map of indices and movie titles
indices = pd.Series(data.index, index=data['title']).drop_duplicates()


def recommend_movies_with_scores(title, cosine_sim=cosine_sim, df=data, indices=indices, top_n=10):
    # Convert the input title to lowercase for case-insensitive matching
    title = title.lower()

    # Checks if the movie is in your dataset
    if title not in df['title'].str.lower().values:
        return f'Movie "{title}" not found in the dataset.'

    # Get the index of the movie that matches the title
    idx = indices[indices.index.str.lower() == title].iloc[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top_n most similar movies
    sim_scores = sim_scores[1: top_n + 1]

    # Get the movie indices and similarity scores
    movie_indices = [(df['title'].iloc[i[0]], i[1]) for i in sim_scores]

    # Return the top_n most similar movies with their similarity scores
    return movie_indices


def main():
    while True:
        # Prompt the user for a movie title
        user_input = input("Enter a movie title (or type 'exit' to quit): ")

        # Exit condition
        if user_input.lower() == 'exit':
            print("Exiting the recommendation system.")
            break

        # Fetch and display recommendations
        try:
            recommendations_with_scores = recommend_movies_with_scores(user_input)
            if recommendations_with_scores:
                print("\nRecommended movies: (The number is the similarity score to the movie you entered)")
                for movie, score in recommendations_with_scores:
                    print(f"{movie}: {score:.3f}")
            else:
                print("No recommendations found.")
        except KeyError:
            print("Movie not found. Please try another title.")
        print("\n")


if __name__ == "__main__":
    main()


