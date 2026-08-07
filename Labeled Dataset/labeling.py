from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

def label_sentences(sentences, summary):
    labels =[]
    for sent in sentences:
        score = scorer.score(summary, sent)["rouge1"].fmeasure
        labels.append(1 if score > 0.2 else 0)
    return labels