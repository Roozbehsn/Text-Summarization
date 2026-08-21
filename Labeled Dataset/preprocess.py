import re
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def clean_text(text):
    
    text = str(text)
    # text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"," ",text)
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
