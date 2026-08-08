import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

def segmentation(text):
  return sent_tokenize(text)