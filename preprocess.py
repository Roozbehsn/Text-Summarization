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
  cleaned_sentences =[]

  for sentence in sentences:
    sentence = sentence.lower()
    sentence = re.sub(r"[^A-Za-z\s]", "", sentence)

    wordTokenize = word_tokenize(sentence)
    cleaned_word =[]
    for word in wordTokenize:
      if  word not in stop_words:
        lemmatization = WordNetLemmatizer().lemmatize(word)
        cleaned_word.append(lemmatization)

    if len(cleaned_word) > 0:
      cleaned_sentences.append(cleaned_word)

  return cleaned_sentences

print (text_preprocess("Hi. This is a test. The best thing"))
 


  

