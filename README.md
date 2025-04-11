# Financial Sentiment Analysis Multilingual Transformer Models

This project is built with the help of Hugging Face's Transformers library.


## About

This repository contains the implementation of transformer-based models for financial sentiment analysis across multiple languages. The project focuses on developing and comparing different model architectures, with an emphasis on knowledge distillation to create more efficient models while maintaining high accuracy.

## Project Status (Updated April 11, 2025)

### Models Trained
The project currently includes several fine-tuned models:

- **Teacher Models**:
  - `bert-base-multilingual-uncased-finetuned-financial-news-sentiment-analysis-european`
  - `xlm-roberta-finetuned-financial-news-sentiment-analysis-european`
  - Language-specific `xlm-roberta-base` models for English, German, Spanish, and French
  - `finbert-european`

- **Student Models**:
  - `distilbert-base-uncased-finetuned-financial-news-sentiment-analysis-european`
  - `distilroberta-finetuned-financial-news-sentiment-analysis-european`

All models are trained for financial sentiment analysis with a focus on European financial news.

### Datasets

- **Financial PhraseBank Multilingual**: Extended to include translations in multiple European languages (English, German, Spanish, French)
- **Synthetic Financial Sentiment**: Generated dataset with multilingual financial sentiment data
- **75% Agreement Subset**: High-confidence subset where at least 75% of annotators agreed on sentiment labels

### Project Structure
- **code/**: Contains core implementation files including data management, model training, distillation, and evaluation
- **data/**: Contains training and evaluation data subsets
- **datasets/**: Houses multilingual datasets and their variants
- **models/**: Stores all trained models with their configurations and checkpoints
- **notebooks/**: Jupyter notebooks for various stages of the project:
  - Dataset translation and preparation
  - Model training and fine-tuning
  - Knowledge distillation
  - Model explainability analysis

### Progress
- Successfully trained multiple teacher models with high accuracy on financial sentiment tasks
- Implemented knowledge distillation to create more efficient student models
- Extended dataset to multiple European languages
- Created synthetic datasets to enhance training
- Conducted model explainability analysis

## Dataset

This project was initially trained on the [Financial Sentiment Analysis dataset](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis) available on Kaggle. The dataset has since been extended with translations and synthetic data to support multilingual capabilities. The datasets contain financial news headlines labeled with sentiment scores, which were used to fine-tune the various models for financial sentiment classification.

## Usage

Check the notebooks directory for examples of how to use the models for financial sentiment analysis tasks. The trained models can be loaded using the Hugging Face Transformers library.

## Requirements

See `requirements.txt` for a list of dependencies needed to run this project.