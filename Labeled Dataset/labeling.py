from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

# def label_sentences(sentences, summary):
#     labels =[]
#     for sent in sentences:
#         score = scorer.score(summary, sent)["rouge1"].fmeasure
#         labels.append(1 if score > 0.2 else 0)
#     return labels


def oracle_label_sentences(sentences,summary,max_sentences=3,min_improvement=0.001):

    labels = [0] * len(sentences)

    selected_indices = []
    # current_summary = ""
    current_score = 0.0

    for _ in range(max_sentences):

        best_index = None
        best_score = current_score

        for i, sentence in enumerate(sentences):

            if i in selected_indices:
                continue

            candidate_indices = selected_indices + [i]
            candidate_indices.sort()

            candidate_summary = " ".join(
                sentences[index]
                for index in candidate_indices
            )
        
            # if current_summary:
            #     candidate_summary = (
            #         current_summary + " " + sentence)
            # else:
            #     candidate_summary = sentence

    
            score = scorer.score(summary, candidate_summary)["rouge1"].fmeasure

    
            if score > best_score + min_improvement:
                best_score = score
                best_index = i

   
        if best_index is None:
            break

        selected_indices.append(best_index)

        labels[best_index] = 1

        # if current_summary:
        #     current_summary += " " + sentences[best_index]
        # else:
        #     current_summary = sentences[best_index]

        current_score = best_score

    return labels