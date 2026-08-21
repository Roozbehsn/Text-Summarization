            
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def train_model(
    bert,
    classifier,
    train_loader,
    val_loader,
    epochs=5,
    learning_rate=5e-6,
    device=None,
    patience=1,
    positive_weight=2.5
):


    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    bert.to(device)
    classifier.to(device)

    class_weights = torch.tensor(
        [1.0, positive_weight],
        dtype=torch.float32,
        device=device
    )


    # Weighted loss for training
    train_criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # Normal loss for validation
    val_criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        list(bert.parameters()) +
        list(classifier.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )


    best_f1 = 0.0
    patience_counter = 0



    for epoch in range(epochs):

        bert.train()
        classifier.train()

        total_train_loss = 0.0


        for input_ids, attention_mask, labels in train_loader:

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)


            optimizer.zero_grad()


            # BERT
            outputs = bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )


            # [CLS] embedding
            cls_embeddings = (
                outputs.last_hidden_state[:, 0, :]
            )


            # Classifier
            logits = classifier(
                cls_embeddings
            )


            # Weighted training loss
            loss = train_criterion(
                logits,
                labels
            )


            # Backpropagation
            loss.backward()


            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(bert.parameters()) +
                list(classifier.parameters()),
                max_norm=1.0
            )


            # Update parameters
            optimizer.step()


            total_train_loss += loss.item()


        average_train_loss = (
            total_train_loss / len(train_loader)
        )




        bert.eval()
        classifier.eval()

        total_val_loss = 0.0

        all_predictions = []
        all_labels = []


        with torch.no_grad():

            for input_ids, attention_mask, labels in val_loader:

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)


                outputs = bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )


                cls_embeddings = (
                    outputs.last_hidden_state[:, 0, :]
                )


                logits = classifier(
                    cls_embeddings
                )


                # Unweighted validation loss
                loss = val_criterion(
                    logits,
                    labels
                )

                total_val_loss += loss.item()


                predictions = torch.argmax(
                    logits,
                    dim=1
                )


                all_predictions.extend(
                    predictions.cpu().tolist()
                )

                all_labels.extend(
                    labels.cpu().tolist()
                )


        average_val_loss = (
            total_val_loss / len(val_loader)
        )


        accuracy = accuracy_score(
            all_labels,
            all_predictions
        )

        precision = precision_score(
            all_labels,
            all_predictions,
            zero_division=0
        )

        recall = recall_score(
            all_labels,
            all_predictions,
            zero_division=0
        )

        f1 = f1_score(
            all_labels,
            all_predictions,
            zero_division=0
        )


        print(
            f"\nEpoch {epoch + 1}/{epochs}"
        )

        print(
            f"Train Loss: {average_train_loss:.4f}"
        )

        print(
            f"Val Loss:   {average_val_loss:.4f}"
        )

        print(
            f"Accuracy:   {accuracy:.4f}"
        )

        print(
            f"Precision:  {precision:.4f}"
        )

        print(
            f"Recall:     {recall:.4f}"
        )

        print(
            f"F1 Score:   {f1:.4f}"
        )

        if f1 > best_f1:

            best_f1 = f1
            patience_counter = 0

            torch.save(
                bert.state_dict(),
                "best_bert_model.pt"
            )

            torch.save(
                classifier.state_dict(),
                "best_classifier.pt"
            )

            print("Best model saved.")

        else:

            patience_counter += 1

        if patience_counter >= patience:

            print("\nEarly stopping.")

            break


    print(
        f"\nBest Validation F1: {best_f1:.4f}"
    )


    return bert, classifier



