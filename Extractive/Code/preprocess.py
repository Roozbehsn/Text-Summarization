import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def text_preprocess(text):
    sentences = sent_tokenize(text)
    output = []

    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r"[^A-Za-z\s]", "", sentence)

        tokens = word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)

        cleaned_sentence = []

        for word, tag in tagged:
            if word not in stop_words:
                lemma = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
                cleaned_sentence.append(lemma)

        sentence_string = " ".join(cleaned_sentence)
        output.append(sentence_string)

    return output 

