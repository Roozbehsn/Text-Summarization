from transformers import AutoModel

def load_bert(model_name="bert-base-uncased"):
    bert = AutoModel.from_pretrained(model_name)
    return bert


def encode_sentences(bert, input_ids, attention_mask):
  
    outputs = bert(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    last_hidden_state = outputs.last_hidden_state

    cls_embeddings = last_hidden_state[:, 0, :]

    return cls_embeddings
