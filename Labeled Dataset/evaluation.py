import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rouge_score import rouge_scorer

def evaluate_classifier(bert,classifier,test_loader,device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert.to(device)
    classifier.to(device)

    bert.eval()
    classifier.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for input_ids , attention_mask , labels in test_loader:

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = bert(input_ids=input_ids,attention_mask=attention_mask)

            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            logits = classifier(cls_embeddings)

            predictions = torch.argmax(logits,dim=1)

            all_predictions.extend(predictions.cpu().tolist())

            all_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(all_labels,all_predictions)

    precision = precision_score(all_labels,all_predictions,zero_division=0)

    recall = recall_score(all_labels,all_predictions,zero_division=0)

    f1 = f1_score(all_labels, all_predictions,zero_division=0)


    print(f"Accuracy :  {accuracy:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall   :  {recall:.4f}")
    print(f"F1 Score :  {f1:.4f}")

    return {"accuracy": accuracy,"precision": precision,"recall": recall,"f1": f1}



def generate_model_summary(original_sentences,sentence_scores,num_sentences=5):

    ranked_sentences = sorted(sentence_scores.items(),key=lambda x: x[1],reverse=True)

    selected_indices = [index for index, score in ranked_sentences[:num_sentences]]

    selected_indices.sort()

    summary = " ".join( original_sentences[index]for index in selected_indices)

    return summary


def calculate_rouge(generated_summary,reference_summary):

    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"],use_stemmer=True)

    scores = scorer.score(reference_summary,generated_summary)


    print(f"ROUGE-1 F1: "f"{scores['rouge1'].fmeasure:.4f}")

    print(f"ROUGE-2 F1: "f"{scores['rouge2'].fmeasure:.4f}")

    print(f"ROUGE-L F1: "f"{scores['rougeL'].fmeasure:.4f}")

    return {"rouge1": scores["rouge1"],"rouge2": scores["rouge2"],"rougeL": scores["rougeL"]}


def calculate_compression_ratio(original_text,summary):

    original_words = len(original_text.split())

    summary_words = len(summary.split())

    if original_words == 0:
        return 0

    compression_ratio = (summary_words / original_words)

    print(f"Original words: {original_words}")

    print(f"Summary words : {summary_words}")

    print(f"Compression ratio: "f"{compression_ratio:.2%}")

    return compression_ratio


def evaluate_summary(original_text,original_sentences,generated_summary,reference_summary):

    print("\nGenerated Summary:")
    print(generated_summary)

    print("\nReference Summary:")
    print(reference_summary)


    rouge_results = calculate_rouge(generated_summary,reference_summary)

    compression_ratio = calculate_compression_ratio(original_text, generated_summary )

    return {"rouge": rouge_results,
            "compression_ratio": compression_ratio}