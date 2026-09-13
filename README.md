# Text Summarization

A text summarization project exploring unsupervised and supervised extractive methods

The **supervised** approach uses BERT sentence embeddings and a trained classifier on the CNN/DailyMail dataset to learn which sentences best represent a document, while the **unsupervised** approach applies TF-IDF sentence representations and the TextRank algorithm to a custom tennis news dataset requiring no labeled data or training. Together, these pipelines compare two fundamentally different philosophies for extractive summarization.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Approach 1: Labeled (Supervised — BERT)](#approach-1-labeled-supervised--bert)
- [Approach 2: Unlabeled (Unsupervised — TF-IDF / TextRank)](#approach-2-unlabeled-unsupervised--tf-idf--textrank)
- [Comparison of Approaches](#comparison-of-approaches)
- [Technologies](#technologies)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Known Limitations & Output Quality](#known-limitations--output-quality)
- [Contributing](#contributing)
- [Contributors](#contributors)
- [License](#license)

## Overview

This project explores automatic summarization through multiple methodologies. Today it covers extractive summarization, selecting the most important sentences from a document to form a summary via two contrasting approaches:

| | Labeled (Supervised) | Unlabeled (Unsupervised) |
|---|---|---|
| **Method** | BERT encoding + trained classifier | TF-IDF + TextRank (graph-based ranking) |
| **Dataset** | CNN/DailyMail | `tennis_articles_v4.csv` |
| **Needs labels?** | Yes | No |
| **Core idea** | Learn which sentences are "summary-worthy" from labeled examples | Rank sentences by similarity/centrality without training |

## Repository Structure

```
.
├── Labeled Dataset/              # Supervised, BERT-based pipeline (CNN/DailyMail)
│   ├── Main.py                   # Entry point — runs the full labeled pipeline
│   ├── preprocess.py             # Cleans and prepares raw articles
│   ├── segmentation.py           # Splits documents into sentences
│   ├── labeling.py               # Generates sentence-level labels (summary / not summary)
│   ├── encoding.py               # BERT-based sentence encoding
│   ├── classifier.py             # Classifier that scores/selects summary sentences
│   ├── training.py                # Model training loop
│   ├── evaluation.py             # ROUGE / accuracy evaluation of generated summaries
│   └── summarize.py              # Produces final extractive summary from predictions
│
└── Unlabeled Dataset/             # Unsupervised, TF-IDF/TextRank pipeline (tennis articles)
    ├── Code/
    │   ├── Main.py                # Entry point — runs the full unlabeled pipeline
    │   ├── preprocess.py          # Cleans and prepares raw articles
    │   ├── sentence_representation.py  # TF-IDF vectorization of sentences
    │   ├── redundancy.py          # Removes/penalizes redundant sentences
    │   ├── textrank.py            # Graph-based sentence ranking (TextRank algorithm)
    │   ├── summary.py             # Assembles the final extractive summary
    │   └── plot.py                # Visualizes results (e.g. sentence scores, graphs)
    └── Dataset/
        └── tennis_articles_v4.csv # Raw tennis news articles used for summarization

```

## Approach 1: Labeled (Supervised — BERT)

Pipeline: `preprocess → segmentation → labeling → encoding → training → classifier → evaluation → summarize`

1. **Preprocess** – Clean and normalize raw CNN/DailyMail articles.
2. **Segmentation** – Split each article into individual sentences.
3. **Labeling** – Assign binary labels to sentences (e.g. based on overlap with reference/highlight summaries).
4. **Encoding** – Generate sentence embeddings using BERT.
5. **Training** – Train a classifier on the labeled sentence embeddings.
6. **Classifier** – Predict summary-worthiness scores for new sentences.
7. **Evaluation** – Score generated summaries against reference summaries (e.g. ROUGE).
8. **Summarize** – Select top-ranked sentences to build the final summary.

## Approach 2: Unlabeled (Unsupervised — TF-IDF / TextRank)

Pipeline: `preprocess → sentence_representation → textrank → redundancy → summary → plot`

1. **Preprocess** – Clean and tokenize tennis articles from `tennis_articles_v4.csv`.
2. **Sentence Representation** – Vectorize sentences using TF-IDF.
3. **TextRank** – Build a sentence similarity graph and rank sentences using the TextRank algorithm (PageRank-style).
4. **Redundancy** – Filter or penalize sentences that are too similar to already-selected ones.
5. **Summary** – Select the top-ranked, non-redundant sentences as the final summary.
6. **Plot** – Visualize sentence scores/graph structure for inspection.

## Comparison of Approaches

The project allows different extractive summarization techniques to be explored and compared.

### Unsupervised Approach

```
TF-IDF
   +
TextRank
   +
Redundancy Reduction
```

**Advantages:**

- Does not require labeled training data
- Uses traditional NLP techniques
- Relatively simple and interpretable
- Useful for exploring graph-based ranking methods

### Supervised Approach

```
BERT
   +
Sentence Classification
   +
MMR
```

**Advantages:**

- Uses contextual representations
- Learns sentence importance from training data
- Can capture semantic information better than traditional TF-IDF representations
- MMR improves diversity and reduces redundant sentences

## Technologies

- **Python 3.x** – core implementation language
- **BERT** (via `transformers`) – contextual sentence embeddings for the supervised approach
- **PyTorch** – model training and inference for the classifier
- **scikit-learn** – TF-IDF vectorization and general ML utilities
- **NetworkX** – graph construction and ranking for TextRank
- **NLTK / spaCy** – sentence segmentation and text preprocessing
- **Pandas / NumPy** – data loading and manipulation
- **Matplotlib** – plotting and visualization of results
- **ROUGE (`rouge-score`)** – evaluation of generated summaries against reference summaries

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Roozbehsn/Text-Summarization.git
   cd Text-Summarization
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install transformers torch scikit-learn nltk spacy pandas numpy networkx matplotlib rouge-score
   ```

4. **Download NLTK/spaCy resources** (if applicable)
   ```bash
   python -m nltk.downloader punkt stopwords
   ```

## Getting Started

### Running the Labeled (BERT) Pipeline
```bash
cd "Labeled Dataset"
python Main.py
```

### Running the Unlabeled (TF-IDF/TextRank) Pipeline
```bash
cd "Unlabeled Dataset/Code"
python Main.py
```

## Known Limitations & Output Quality

The results shown by this project especially from the **supervised, BERT-based approach** are directly influenced by the hardware they run on. BERT encoding and classifier training are computationally intensive, and running them on a machine with more powerful hardware (e.g. a dedicated GPU with more VRAM, or a multi-core CPU with more RAM) will noticeably improve both **output quality and runtime efficiency**.

On limited hardware (e.g. CPU-only environments), the supervised pipeline may need to use smaller batch sizes, fewer training epochs, or a lighter BERT variant (e.g. `bert-base` instead of `bert-large`, or a distilled model like `distilbert`) to keep runtimes reasonable  which can come at some cost to summary quality. The unsupervised TF-IDF/TextRank pipeline is comparatively lightweight and less sensitive to hardware constraints, since it doesn't involve training a neural network.

If you have access to a GPU (locally or via a cloud service such as Google Colab, Kaggle Notebooks, or AWS/GCP), it's recommended for running and experimenting with the supervised approach, particularly during training.

## Contributing

Contributions are welcome! To contribute:

1. **Fork** the repository.
2. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**, following the existing code style and structure.
4. **Test your changes** to make sure both pipelines still run end-to-end.
5. **Commit and push**:
   ```bash
   git commit -m "Add: brief description of your change"
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request** describing what you changed and why.

You can also reach out via email at rseyednozadi@gmail.com.

## Contributors
 
- [@homaajam](https://github.com/homaajam)
- [@Roozbehsn](https://github.com/Roozbehsn)

## License

This project is licensed under the [MIT License](LICENSE) — see the `LICENSE` file for details. The original paper referenced in this repository remains the intellectual property of its respective authors and is included here for educational/reference purposes only.
