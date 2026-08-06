from nltk.corpus import gutenberg
from preprocess import text_preprocess
from sentence_representation import graph
from sentence_representation import sentence_representation
from textrank import textrank
from plot import visualize_similarity_matrix
from plot import visualize_graph
from redundancy import remove_redundancy
from summary import generate_summary
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize

#Other Dataset you can use : "blake-poems.txt" , "carroll-alice.txt" , "whitman-leaves.txt"
text = pd.read_csv("tennis_articles_v4.csv")
text=text["article_text"][1]

original_sentences = sent_tokenize(text)
preprocessed_text = text_preprocess(text)

text_sentences = sentence_representation(preprocessed_text)
visualize_similarity_matrix(text_sentences)

text_graph = graph(text_sentences)

textrank_text = textrank(text_graph)
visualize_graph(text_graph,textrank_text)

selected_indices = remove_redundancy(
    textrank_text,
    text_sentences,
    num_sentences=5,
    similarity_threshold=0.7
)

text_summary = generate_summary(original_sentences, selected_indices)

print(text)
print("----------------------------------------------")
print(text_summary)
