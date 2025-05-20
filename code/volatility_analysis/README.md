# Volatility Analysis Library

This library provides tools for analyzing financial market volatility in relation to news sentiment. It includes:

- Sentiment analysis of financial news using pre-trained models
- LSTM-based volatility prediction
- Data preparation and model evaluation utilities
- Customizable visualization functions

## Features

- Analyze the relationship between news sentiment and market volatility
- Predict future volatility using LSTM neural networks
- Create high-quality visualizations of financial data
- Support for different market indices (FTSE, IBEX, DAX, CAC, etc.)

## Usage Example

```python
from code.volatility_analysis import run_volatility_pipeline

# Run the complete volatility analysis pipeline
results = run_volatility_pipeline(
    news_df=news_df,              # DataFrame with news data
    stock_data=market_data,       # DataFrame with stock market data
    market_name="FTSE 100",       # Name of the market index
    cut_date="2023-12-01",        # Date to split train/test data
    output_dir="../news",         # Directory for output files
    seq_len=10,                   # Sequence length for LSTM
    epochs=50,                    # Number of training epochs
    batch_size=16,                # Batch size for training
    learning_rate=0.001,          # Learning rate for optimizer
    verbose=True                  # Print progress information
)
```

## Library Structure

The library provides the following main functions:

### Data Processing Functions

- `initialize_sentiment_model`: Initialize the sentiment analysis model
- `infer_sentiment`: Analyze sentiment of a given text
- `calculate_sentiment`: Calculate sentiment for a specific date
- `split_data`: Split data into training and testing sets
- `prepare_lstm_data`: Prepare data for LSTM model training
- `create_sequences`: Create input sequences for LSTM

### Model Functions

- `LSTMVolatility`: LSTM model class for volatility prediction
- `train_lstm_model`: Train the LSTM model
- `evaluate_lstm_model`: Evaluate model performance

### Visualization Functions

- `plot_volatility_news_count`: Plot volatility and news article counts
- `plot_volatility_sentiment`: Plot volatility with sentiment markers
- `plot_prediction_results`: Plot actual vs. predicted volatility

### Pipeline Function

- `run_volatility_pipeline`: Run the entire analysis pipeline

## Example Notebooks

Example usage of the library can be found in these notebooks:

- `5_volatility_analysis_ftse_using_lib.ipynb`: FTSE 100 analysis
- `5_volatility_analysis_ibex_using_lib.ipynb`: IBEX 35 analysis
- `5_volatility_analysis_dax_using_lib.ipynb`: DAX analysis
- `5_volatility_analysis_cac_using_lib.ipynb`: CAC 40 analysis

## Requirements

- Python 3.6+
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Transformers
- scikit-learn
- yfinance (for market data)
- pygooglenews (for news data)

## Authors

DistilBERT Financial Sentiment Team
