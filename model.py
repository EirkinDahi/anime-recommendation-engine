import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

df = pd.read_csv("Data/anime.csv")
df= df.reset_index()
#df.head()

# df.columns
features = ["Score","Rating","Producers","Ranked","Plan to Watch","Popularity","Type","Genres"]

def combine_features(r):
    return str(r['Score']) + " " + str(r['Rating']) + " " + r['Producers'] + " " + str(r['Ranked']) + " " + str(r['Plan to Watch']) + " " +str(r['Popularity']) + " " +r['Type'] + " " + r['Genres']

for feature in features:
    df[feature] = df[feature].fillna("")
df["combined_features"] = df.apply(combine_features,axis =1)

# print(df.head())

cv_anime = CountVectorizer()
count_matrix_anime = cv_anime.fit_transform(df["combined_features"])

cosine_similarity_anime = cosine_similarity(count_matrix_anime)

def get_id_from_title(name):
    return df[df.Name == name]["index"].values[0]

def get_title_from_id(id):
    return df[df.index == id]["Name"].values[0]

def results(anime_name):
    anime_index = get_id_from_title(anime_name)
    similar_animes = list(enumerate(cosine_similarity_anime[anime_index]))
    sorted_anime_recommendations = sorted(similar_animes,key=lambda x:x[1], reverse = True)[1:]
    top_ten = []

    for i in range(10):
        top_ten.append(get_title_from_id(sorted_anime_recommendations[i][0]))
    return top_ten


# print(results("Naruto"))

# anime_user = input("Enter an anime: ")
# anime_index = get_id_from_title(anime_user)
# similar_animes = list(enumerate(cosine_similarity_anime[anime_index]))

# sorted_anime_recommendations = sorted(similar_animes,key=lambda x:x[1], reverse = True)[1:]
# print(sorted_anime_recommendations)

# print("\nTop 10 Recommendations for " + anime_user + " :")
# for i in range(10):
#     print(get_title_from_id(sorted_anime_recommendations[i][0]))

# joblib.dump(sorted_anime_recommendations, 'model.pkl')

# sorted_anime_recommendations = joblib.load('model.pkl')