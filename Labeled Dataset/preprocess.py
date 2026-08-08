import re
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def clean_text(text):
    
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def preprocess_sentences(sentences):
  
    cleaned_sentences = []

    for sentence in sentences:

        sentence = clean_text(sentence)
        if sentence:
            cleaned_sentences.append(sentence)

    return cleaned_sentences


def tokenize_sentences(sentences, max_length=128):
   
    encoded = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return encoded


def prepare_training_example(sentences , labels, max_length=128):
  
    if len(sentences) != len(labels):
        raise ValueError(
            "Number of sentences and labels must be the same."
        )

    cleaned_sentences = preprocess_sentences(sentences)

    if len(cleaned_sentences) != len(labels):
        raise ValueError(
            "Sentence count changed during preprocessing. "
            "Labels are no longer aligned."
        )
    encoded = tokenize_sentences(
        cleaned_sentences,
        max_length=max_length
    )

    encoded["labels"] = labels

    encoded["sentences"] = cleaned_sentences

    return encoded