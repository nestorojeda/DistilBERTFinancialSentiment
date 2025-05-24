# Financial Sentiment Analysis & Market Volatility Prediction

This project combines multilingual transformer models for financial sentiment analysis with market volatility prediction. Built with Hugging Face's Transformers library and PyTorch.

## About

This repository offers a comprehensive toolkit for financial market analysis, with two main components:

1. **Sentiment Analysis**: Implementation of transformer-based models for financial sentiment analysis across multiple European languages
2. **Volatility Analysis**: Tools for analyzing and predicting market volatility using sentiment data and LSTM-based deep learning models

The project emphasizes both accuracy and efficiency through knowledge distillation techniques and a modular, reusable architecture.

## Project Status (Updated May 20, 2025)

### Sentiment Analysis Models

The project includes several fine-tuned models for financial sentiment analysis:

- **Teacher Models**:
  - `bert-base-multilingual-uncased-finetuned-financial-news-sentiment-analysis-european`
  - `xlm-roberta-finetuned-financial-news-sentiment-analysis-european`
  - Language-specific `xlm-roberta-base` models for English, German, Spanish, and French
  - `finbert-european`

- **Student Models**:
  - `distilbert-base-uncased-finetuned-financial-news-sentiment-analysis-european`
  - `distilroberta-finetuned-financial-news-sentiment-analysis-european`

All models are trained for financial sentiment analysis with a focus on European financial news.

### Volatility Analysis Library

A new addition to the project is the volatility analysis library (`code/volatility_analysis/`), which provides:

- Sentiment analysis of financial news using the pre-trained models
- LSTM-based volatility prediction
- Data preparation and model evaluation utilities
- Customizable visualization functions
- Full pipeline for analyzing market indices

Currently supports analysis for:
- FTSE 100 (UK)
- IBEX 35 (Spain)
- DAX 40 (Germany)
- CAC 40 (France)

### Datasets

- **Financial PhraseBank Multilingual**: Extended to include translations in multiple European languages (English, German, Spanish, French)
- **Synthetic Financial Sentiment**: Generated dataset with multilingual financial sentiment data
- **75% Agreement Subset**: High-confidence subset where at least 75% of annotators agreed on sentiment labels
- **Financial News Data**: Collected financial news across European markets (2020-2025) for sentiment and volatility analysis

### Project Structure
- **code/**: Contains core implementation files
  - **distillation/**: Knowledge distillation implementation
  - **finetuning/**: Model fine-tuning utilities
  - **news_scrapper/**: Tools for collecting financial news data
  - **volatility_analysis/**: Library for market volatility analysis and prediction
- **data/**: Contains training and evaluation data subsets
- **datasets/**: Houses multilingual datasets and their variants
- **models/**: Stores all trained models with their configurations and checkpoints
- **news/**: Financial news data and generated visualizations by market
- **notebooks/**: Jupyter notebooks for various stages of the project:
  - Dataset translation and preparation (1_*.ipynb)
  - Model training and fine-tuning (2_*.ipynb, 3_*.ipynb)
  - Model evaluation and analysis (4_*.ipynb)
  - Volatility analysis by market (5_*.ipynb)

### Recent Developments
- **NEW**: Developed a reusable volatility analysis library for financial markets
- **NEW**: Created standardized notebooks for analyzing FTSE, IBEX, DAX, and CAC indices
- **NEW**: Implemented LSTM-based market volatility prediction using sentiment data
- Successfully trained multiple teacher models with high accuracy on financial sentiment tasks
- Implemented knowledge distillation to create efficient student models
- Extended datasets to multiple European languages
- Created synthetic datasets to enhance training

## Datasets

- **Sentiment Analysis**: Initially trained on the [Financial Sentiment Analysis dataset](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis) from Kaggle. Extended with translations and synthetic data to support multilingual capabilities.

- **Volatility Analysis**: Uses financial news collected through Google News API for various European markets (FTSE, IBEX, DAX, CAC) combined with market data from Yahoo Finance.

## Usage

### Sentiment Analysis
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load pre-trained model and tokenizer
model_name = "nojedag/xlm-roberta-finetuned-financial-news-sentiment-analysis-european"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Analyze sentiment
text = "Company reports strong Q1 earnings exceeding market expectations"
inputs = tokenizer(text, return_tensors="pt", truncation=True)
outputs = model(**inputs)
sentiment = outputs.logits.argmax(dim=1).item()
# Map to -1 (negative), 0 (neutral), 1 (positive)
```

### Volatility Analysis
```python
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'code', 'volatility_analysis'))
from volatility_lib import *

# Run the volatility analysis pipeline with sentiment
results = run_volatility_pipeline(
    news_df=news_df,                # DataFrame with news data
    stock_data=market_data,         # DataFrame with stock market data
    market_name="FTSE 100",         # Market index name
    cut_date="2023-12-01",          # Train/test split date
    output_dir="./news",            # Output directory
    epochs=50,                      # Training epochs
    use_sentiment=True,             # Use sentiment inference (default)
    verbose=True                    # Show progress
)

# Run volatility analysis without sentiment (volatility-only model)
results_no_sentiment = run_volatility_pipeline(
    news_df=news_df,
    stock_data=market_data,
    market_name="FTSE 100",
    cut_date="2023-12-01",
    output_dir="./news",
    epochs=50,
    use_sentiment=False,            # Skip sentiment - use only volatility history
    verbose=True
)
```

Check the `notebooks` directory for complete examples of sentiment analysis and volatility prediction.

## Requirements

See `requirements.txt` for a complete list of dependencies. Major dependencies include:

- PyTorch
- Transformers (Hugging Face)
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- yfinance

## Future Work

- Expansion to more markets and languages
- Integration with real-time market data
- Enhanced visualization dashboard
- Incorporation of additional financial indicators