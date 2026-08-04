from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def sentence_representation(tokenized_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tokenized_text)
    sentence_vectors = tfidf_matrix.toarray()
    similarity_matrix = cosine_similarity(sentence_vectors)
    return similarity_matrix

def graph(similarity_matrix, threshold=0.1):
    output = nx.Graph()
    
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix)):
            if i != j and similarity_matrix[i][j] > threshold:
                output.add_edge(i, j, weight=similarity_matrix[i][j])
    return output
    
