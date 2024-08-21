# News Categorization and Sentiment Analysis using Fine-Tuned Pretrained Models

This project focuses on classifying news articles into various categories and analyzing the sentiment of the content using fine-tuned pretrained language models like BERT, DistilBERT, and RoBERTa. The dataset was sourced from the News API and underwent extensive preprocessing before model training.



## Project Overview

This project aims to develop an effective news categorization system by fine-tuning pretrained language models on a custom dataset. Additionally, it includes sentiment analysis to gauge public sentiment surrounding different news categories.

## Dataset

The dataset was sourced from [News API](https://newsapi.org/), containing articles categorized into various topics such as politics, technology, sports, entertainment, etc. The dataset includes the following columns:

- `source`
- `author`
- `title`
- `description`
- `url`
- `urlToImage`
- `publishedAt`
- `content`
- `category` (target column for classification)

## Preprocessing

Before feeding the data into the models, the following preprocessing steps were performed:

1. **Stopwords Removal:** Commonly used words (stopwords) were removed to reduce noise in the text data.
2. **Punctuation Removal:** All punctuation marks were stripped from the text to ensure uniformity.
3. **Text Normalization:** The text was converted to lowercase, and extra whitespace was removed.
4. **Tokenization and Padding:** The text was tokenized and padded to a uniform length suitable for model input.

The preprocessing was done using libraries such as NLTK, SpaCy, and Transformers from Hugging Face.

## Models

The following pretrained models were fine-tuned for the task:

- **BERT (Bidirectional Encoder Representations from Transformers):** A powerful transformer-based model for NLP tasks.
- **DistilBERT:** A distilled version of BERT, offering a faster and lighter model with competitive performance.
- **RoBERTa (Robustly optimized BERT approach):** An optimized version of BERT with improvements in training methodology.

## Fine-Tuning

The models were fine-tuned on the preprocessed dataset with the following steps:

1. **Model Initialization:** Pretrained models were loaded using the Hugging Face Transformers library.
2. **Training:** The models were fine-tuned on the dataset using the category column as the target label.
3. **Hyperparameter Tuning:** Key hyperparameters such as learning rate, batch size, and number of epochs were tuned to optimize performance.
4. **Sentiment Analysis:** A secondary task of sentiment analysis was added to categorize the articles' sentiment as positive, negative, or neutral.

## Evaluation

The models were evaluated using the following metrics:

- **Accuracy:** The overall accuracy of the model in classifying news categories.
- **F1 Score, Precision, Recall:** Metrics to evaluate the model’s performance on imbalanced classes.
- **Confusion Matrix:** To visualize the performance of the classification tasks.
- **Sentiment Accuracy:** To measure the effectiveness of the sentiment analysis.

