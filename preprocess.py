import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def text_preprocess(text):
  sentences = sent_tokenize(text)
  output=[] 
  for sentence in sentences:
    sentence = sentence.lower()
    sentence = re.sub(r"[^A-Za-z\s]", "", sentence)
    wordTokenize = word_tokenize(sentence)
    cleaned_sentences =[]
    for word in wordTokenize:
      if  word not in stop_words:
        lemmatization = WordNetLemmatizer().lemmatize(word)
        cleaned_sentences.append(lemmatization)
              
    Sentence_string=(" ".join(cleaned_sentences))
    output.append(Sentence_string)

  return output

 


  

