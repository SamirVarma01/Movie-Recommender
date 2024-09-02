from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ["London Paris London", "Paris Paris London"]

cv = CountVectorizer() #converts text to numerical values
count_matrix = cv.fit_transform(text) #converts to points depending on count of words (alphabetical)

similarity_scores = cosine_similarity(count_matrix)

print(similarity_scores)