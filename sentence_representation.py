from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def sentence_representation(tokenized_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tokenized_text)
    sentence_vectors = tfidf_matrix.toarray()
    similarity_matrix = cosine_similarity(sentence_vectors)
    return similarity_matrix
