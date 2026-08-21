
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

from segmentation import segmentation
from preprocess import preprocess_sentences, tokenize_sentences
from labeling import oracle_label_sentences
from encoding import load_bert
from classifier import SentenceClassifier
from training import train_model
from evaluation import (
    evaluate_classifier,
    generate_model_summary,
    evaluate_summary
)
# from summarize import select_top_k
from summarize import select_mmr


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)



train_articles = pd.read_csv("train.csv")
val_articles = pd.read_csv("validation.csv")
test_articles = pd.read_csv("test.csv")


train_articles = train_articles.iloc[:1000].copy()
val_articles = val_articles.iloc[:200].copy()
test_articles = test_articles.iloc[:200].copy()


train_articles = train_articles[["article", "highlights"]]
val_articles = val_articles[["article", "highlights"]]
test_articles = test_articles[["article", "highlights"]]

def prepare_dataset(articles):

    all_sentences = []
    all_labels = []

    for _ , example in  articles.iterrows():

        article = example["article"]
        reference_summary = example["highlights"]



        sentences = segmentation(article)



        cleaned_sentences = preprocess_sentences(
            sentences
        )


        labels = oracle_label_sentences(
            cleaned_sentences,
            reference_summary ,
            max_sentences=3
        )

   

        all_sentences.extend(cleaned_sentences)
        all_labels.extend(labels)



    encoded = tokenize_sentences(
        all_sentences,
        max_length=128
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    labels = torch.tensor(
        all_labels,
        dtype=torch.long
    )

    return TensorDataset(
        input_ids,
        attention_mask,
        labels
    )


print("\nPreparing training data...")

train_dataset = prepare_dataset(
    train_articles
)
train_labels = train_dataset.tensors[2]

negative_count = (
    train_labels == 0
).sum().item()

positive_count = (
    train_labels == 1
).sum().item()


print("\nClass distribution:")
print(
    "Negative sentences:",
    negative_count
)

print(
    "Positive sentences:",
    positive_count
)


print("Preparing validation data...")



val_dataset = prepare_dataset(
    val_articles
)

print("Preparing test data...")

test_dataset = prepare_dataset(
    test_articles
)


print("\nTraining sentences:", len(train_dataset))
print("Validation sentences:", len(val_dataset))
print("Test sentences:", len(test_dataset))


BATCH_SIZE = 16


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nLoading BERT...")

bert = load_bert(
    "bert-base-uncased"
)


classifier = SentenceClassifier(
    embedding_dim=768
)


print("\nStarting training...")

bert, classifier = train_model(
    bert=bert,
    classifier=classifier,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=5 ,
    learning_rate=5e-6,
    device=device,
    patience=1,
    positive_weight=2.5
)


print("\nLoading best model...")

bert.load_state_dict(
    torch.load(
        "best_bert_model.pt",
        map_location=device ,
        weights_only=True
    )
)

classifier.load_state_dict(
    torch.load(
        "best_classifier.pt",
        map_location=device ,
        weights_only=True
    )
)


print("\n================================")
print("CLASSIFIER EVALUATION")
print("================================")

classifier_results = evaluate_classifier(
    bert=bert,
    classifier=classifier,
    test_loader=test_loader,
    device=device
)



print("\n================================")
print("GENERATING SUMMARY")
print("================================")


test_example = test_articles.iloc[0]

original_text = test_example["article"]

reference_summary = test_example["highlights"]




original_sentences = segmentation(
    original_text
)



cleaned_sentences = preprocess_sentences(
    original_sentences
)


encoded = tokenize_sentences(
    cleaned_sentences,
    max_length=128
)


input_ids = encoded["input_ids"].to(device)

attention_mask = encoded["attention_mask"].to(device)

bert.eval()
classifier.eval()


with torch.no_grad():

    embeddings = bert(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    cls_embeddings = embeddings.last_hidden_state[:, 0, :]


with torch.no_grad():

    logits = classifier(
        cls_embeddings
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    # Probability of sentence being selected
    sentence_scores = probabilities[:, 1]

# selected_sentences = select_top_k(
#     original_sentences,
#     sentence_scores,
#     k=3
# )
selected_sentences = select_mmr(
    sentences=cleaned_sentences,
    scores=sentence_scores,
    embeddings=cls_embeddings,
    k=3,
    lambda_param=0.7
)


generated_summary = " ".join(
    selected_sentences
)



summary_results = evaluate_summary(
    original_text=original_text,
    original_sentences=original_sentences,
    generated_summary=generated_summary,
    reference_summary=reference_summary
)


print("\n================================")
print("FINAL SUMMARY")
print("================================")

print(generated_summary)

print("\n================================")
print("REFERENCE SUMMARY")
print("================================")

print(reference_summary)

print("\n================================")
print("RESULTS")
print("================================")

print("\nClassifier results:")
print(classifier_results)

print("\nSummary results:")
print(summary_results)
