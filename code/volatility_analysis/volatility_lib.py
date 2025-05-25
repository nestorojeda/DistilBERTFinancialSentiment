"""
Volatility Analysis Library for Financial Market Data

This module provides functions and classes for analyzing financial market volatility
in relation to news sentiment. It includes tools for sentiment analysis of financial news,
LSTM-based volatility prediction, and visualization.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any


# Initialize sentiment analysis model and tokenizer
def initialize_sentiment_model(model_name: str = "nojedag/xlm-roberta-finetuned-financial-news-sentiment-analysis-european") -> Tuple:
    """
    Initialize the sentiment analysis model and tokenizer.
    
    Args:
        model_name: The name or path of the pre-trained model to use.
    
    Returns:
        Tuple containing the tokenizer and model objects.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


# Function to infer sentiment from text
def infer_sentiment(text: str, tokenizer: Any, model: Any) -> int:
    """
    Infer sentiment from a given text.
    
    Args:
        text: The text to analyze.
        tokenizer: The tokenizer to use for preprocessing.
        model: The model to use for sentiment analysis.
    
    Returns:
        An integer representing sentiment: -1 (negative), 0 (neutral), or 1 (positive).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = logits.softmax(dim=1)
    sentiment = probabilities.argmax(dim=1).item()
    # Map sentiment values (0, 1, 2) to (-1, 0, 1)
    if sentiment == 2:
        sentiment = -1
    return sentiment


# Get titles for a specific date
def get_titles(date: datetime, news_df: pd.DataFrame) -> List[str]:
    """
    Get news titles for a specific date from a news DataFrame.
    
    Args:
        date: The date to get titles for.
        news_df: DataFrame containing news data with 'date' and 'title' columns.
    
    Returns:
        List of news article titles for the given date.
    """
    return news_df[news_df['date'].dt.date == date.date()]['title'].tolist()


# Calculate sentiment for a specific date
def calculate_sentiment(date: datetime, news_df: pd.DataFrame, tokenizer: Any, model: Any, verbose: bool = False) -> Optional[float]:
    """
    Calculate average sentiment for all news articles on a given date.
    
    Args:
        date: The date to calculate sentiment for.
        news_df: DataFrame containing news data.
        tokenizer: The tokenizer to use for sentiment analysis.
        model: The model to use for sentiment analysis.
        verbose: Whether to print detailed sentiment information.
    
    Returns:
        Average sentiment score or None if no news articles for the date.
    """
    titles = get_titles(date, news_df)
    if not titles:
        return None
    sentiments = [infer_sentiment(title, tokenizer, model) for title in titles]
    return sum(sentiments) / len(sentiments)


# Split data into training and testing sets
def split_data(merged_df: pd.DataFrame, cut_date: str, val_ratio: float = 0.2 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets based on a cut date.
    
    Args:
        merged_df: DataFrame containing merged news and volatility data.
        cut_date: Date string to split the data on (format: 'YYYY-MM-DD').
    
    Returns:
        Tuple of (training_data, testing_data)
    """    # Fill NaN values in sentiment features with 0 (neutral sentiment)
    # and technical indicators with their median values
    for col in merged_df.columns:
        if col.startswith('sentiment'):
            merged_df[col] = merged_df[col].fillna(0)
        elif col != 'date' and merged_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # For numeric columns other than date, fill with median
            merged_df[col] = merged_df[col].fillna(merged_df[col].median())
    
    train = merged_df[merged_df['date'] < cut_date]
    test = merged_df[merged_df['date'] >= cut_date]

    val_size = int(len(train) * val_ratio)
    train = train[:-val_size]
    val = train[-val_size:]

    return train, test, val


# Prepare data for LSTM model
def prepare_lstm_data(train: pd.DataFrame, test: pd.DataFrame, val: pd.DataFrame,
                     feature_cols: List[str] = ['Volatility_Smooth', 'sentiment'], 
                     target_col: str = 'Volatility_Smooth', 
                     seq_len: int = 10) -> Tuple:
    """
    Prepare data for LSTM model training and testing.
    
    Args:
        train: Training data DataFrame.
        test: Testing data DataFrame.
        feature_cols: List of feature column names.
        target_col: Name of target column.
        seq_len: Sequence length for LSTM.
    
    Returns:
        Tuple containing prepared data and scalers.
    """    # Verify all required columns exist in both dataframes
    missing_train = [col for col in feature_cols if col not in train.columns]
    missing_test = [col for col in feature_cols if col not in test.columns]
    
    if missing_train or missing_test:
        print(f"Warning: Missing columns in train: {missing_train}, test: {missing_test}")
        # Add missing columns with zeros
        for col in missing_train:
            train[col] = 0
        for col in missing_test:
            test[col] = 0
    
    # Select features and target
    train_data = train[feature_cols + [target_col]].copy()
    test_data = test[feature_cols + [target_col]].copy()
    val_data = val[feature_cols + [target_col]].copy() if not val.empty else pd.DataFrame(columns=feature_cols + [target_col])
    
    # Normalize features and target
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_y_single = MinMaxScaler()  # For single column output

    X_train = scaler_x.fit_transform(train_data[feature_cols])
    y_train = scaler_y.fit_transform(train_data[[target_col]])\
        
    # Fit the single-column scaler on the target column values as a 1D array
    scaler_y_single.fit(train_data[[target_col]].values.reshape(-1, 1))

    X_test = scaler_x.transform(test_data[feature_cols])
    y_test = scaler_y.transform(test_data[[target_col]])

    X_val = scaler_x.transform(val_data[feature_cols])
    y_val = scaler_y.transform(val_data[[target_col]])

    # Create sequences for LSTM
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_len)

    # Convert to torch tensors
    X_train_seq = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_seq = torch.tensor(y_train_seq, dtype=torch.float32)
    X_test_seq = torch.tensor(X_test_seq, dtype=torch.float32)
    y_test_seq = torch.tensor(y_test_seq, dtype=torch.float32)
    X_val_seq = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_seq = torch.tensor(y_val_seq, dtype=torch.float32)

    return (X_train_seq, y_train_seq, X_test_seq, y_test_seq, X_val_seq, y_val_seq,
            scaler_x, scaler_y, scaler_y_single)


# Create sequences for LSTM
def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences and corresponding output values for LSTM training.
    
    Args:
        X: Input features array.
        y: Target values array.
        seq_length: Length of each sequence.
    
    Returns:
        Tuple of (input_sequences, output_values)
    """
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(xs), np.array(ys)


# LSTM Model for Volatility Prediction with Attention and Dropout
class ImprovedLSTMVolatility(nn.Module):
    """Enhanced LSTM with attention and dropout."""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2):
        super(ImprovedLSTMVolatility, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)  # Residual connection
        
        # Take the last output
        out = attn_out[:, -1, :]
        out = self.dropout(out)
        
        # Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


# Evaluate LSTM model
def evaluate_lstm_model(model: nn.Module, X_test_seq: torch.Tensor, y_test_seq: torch.Tensor, 
                       scaler_y: MinMaxScaler) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Evaluate an LSTM model and calculate performance metrics.
    
    Args:
        model: Trained LSTM model.
        X_test_seq: Input sequences for testing.
        y_test_seq: Target values for testing.
        scaler_y: Scaler used to normalize target values.
    
    Returns:
        Tuple of (predictions, actual_values, metrics_dict)
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_seq).numpy()
        # Use the original scaler for both predictions and test data
        y_pred_inv = scaler_y.inverse_transform(y_pred)[:, 0:1]  # Extract the volatility column
        y_test_inv = scaler_y.inverse_transform(y_test_seq.numpy())[:, 0:1]  # Extract the volatility column

    # Calculate error metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print(f"Model Performance Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    return y_pred_inv, y_test_inv, metrics


# Plot volatility and news count
def plot_volatility_news_count(merged_df: pd.DataFrame, 
                              market_name: str = 'Market',
                              save_path: Optional[str] = None,
                              show_plot: bool = True) -> None:
    """
    Plot volatility and news article counts.
    
    Args:
        merged_df: DataFrame with merged volatility and news data.
        market_name: Name of the market for plot titles and labels.
        save_path: Path to save the plot image, or None to skip saving.
        show_plot: Whether to display the plot.
    """
    # Set the style with improved aesthetics
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

    # Create figure and axes with better size
    fig, ax1 = plt.subplots(figsize=(16, 8), dpi=100)
    ax2 = ax1.twinx()  # Create a twin y-axis for volatility

    # Calculate better marker scaling to avoid extremely large markers
    max_count = merged_df['count'].max()
    min_size = 30
    max_size = 200
    size_scale = (max_size - min_size) / max_count if max_count > 0 else 1

    # Plot news article counts (left y-axis, as blue scatter)
    scatter = ax1.scatter(merged_df['date'], merged_df['count'], 
                        s=merged_df['count']*size_scale + min_size,  # Better sizing formula
                        color='#2176ae', 
                        alpha=0.7, 
                        edgecolor='white', 
                        linewidth=1.5, 
                        label='News Article Count', 
                        zorder=3)

    # Volatility line (right y-axis)
    sns.lineplot(x='date', y='Volatility_Smooth', 
               data=merged_df, 
               ax=ax2, 
               color='#d7263d', 
               label=f'{market_name} Volatility', 
               linewidth=3, 
               zorder=2)

    # Improved y-axis formatting
    ax1.set_ylabel("News Article Count", fontsize=16, color='#2176ae', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#2176ae', labelsize=14)
    ax1.set_xlabel("Date", fontsize=16, fontweight='bold')
    ax1.set_ylim(0, merged_df['count'].max() * 1.2)  # Set reasonable y-axis limits

    # Format right y-axis (volatility)
    ax2.set_ylabel(f"{market_name} Volatility", fontsize=16, color='#d7263d', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#d7263d', labelsize=14)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=8))

    # Add grid for better readability
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # Annotate article counts more elegantly
    for i, row in merged_df.iterrows():
        if row['count'] > 0:
            # Only annotate counts above a threshold to avoid cluttering
            if row['count'] >= max(1, max_count * 0.1):  # Annotate counts that are at least 10% of max count
                ax1.text(row['date'], row['count'] + max_count * 0.05,  # Position slightly above point
                        str(row['count']), 
                        color='#1b2a41', 
                        ha='center', 
                        va='bottom', 
                        fontsize=11, 
                        fontweight='bold', 
                        alpha=0.9)

    # Improved X-axis formatting with proper date range
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Include year in format
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=12)  # Rotate labels for better readability

    # Calculate the date range for dynamic title
    start_year = merged_df['date'].min().year
    end_year = merged_df['date'].max().year
    year_range = f"{start_year}" if start_year == end_year else f"{start_year}-{end_year}"

    # Title and legend with dynamic year range
    plt.title(f"{market_name} Volatility and News Article Counts ({year_range})", 
             fontsize=20, fontweight='bold', pad=20)

    # Improved legend positioning and formatting
    ax1.legend(loc='upper left', fontsize=14, frameon=True, fancybox=True, borderpad=1)
    ax2.legend(loc='upper right', fontsize=14, frameon=True, fancybox=True, borderpad=1)

    # Add a subtle background color for better contrast
    fig.patch.set_facecolor('#f8f9fa')

    # Add tight layout and adjust to prevent any overlapping elements
    plt.tight_layout(pad=3)

    # Add a subtle grid background for the entire plot
    ax1.grid(True, linestyle='--', alpha=0.2)

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show the plot
    if show_plot:
        plt.show()
    else:
        plt.close()


# Plot volatility and sentiment
def plot_volatility_sentiment(merged_df: pd.DataFrame, 
                             market_name: str = 'Market',
                             save_path: Optional[str] = None,
                             show_plot: bool = True) -> None:
    """
    Plot volatility and news sentiment.
    
    Args:
        merged_df: DataFrame with merged volatility and sentiment data.
        market_name: Name of the market for plot titles and labels.
        save_path: Path to save the plot image, or None to skip saving.
        show_plot: Whether to display the plot.
    """
    # Define sentiment colors and labels
    label_map = {"negative": -1, "neutral": 0, "positive": 1}
    sentiment_colors = {-1: "#d7263d", 0: "#ffd166", 1: "#06d6a0"}  # Red for negative, yellow for neutral, green for positive
    inverse_label_map = {v: k for k, v in label_map.items()}

    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()  # Create a second y-axis

    # Plot volatility line
    ax2.plot(merged_df['date'], merged_df['Volatility_Smooth'], color='#2176ae', linewidth=2.5, 
             label=f'{market_name} Volatility', alpha=0.8)

    # Plot sentiment markers on the volatility line
    for i, row in merged_df.iterrows():
        # Skip rows with NaN sentiment values
        if pd.isna(row['sentiment']):
            continue
        else:
            sentiment_value = int(row['sentiment'])
            # Plot sentiment markers directly on the volatility line
            if (sentiment_value > -2) and (sentiment_value < 2):
                # Use the sentiment value to determine the color
                ax2.scatter(row['date'], row['Volatility_Smooth'], 
                            color=sentiment_colors[sentiment_value], 
                            s=100, 
                            zorder=5,
                            edgecolor='white', 
                            linewidth=1.5)

    # Add colored marker samples for the legend
    for sentiment_value in [-1, 0, 1]:
        ax1.scatter([], [], color=sentiment_colors[sentiment_value], 
                    label=f"{inverse_label_map[sentiment_value].capitalize()} Sentiment",
                    s=100, edgecolor='white', linewidth=1.5)

    # Format axes
    ax1.set_xlabel("Date", fontsize=15, fontweight='bold')
    ax2.set_ylabel("Volatility", fontsize=15, fontweight='bold', color='#2176ae')

    # Hide the y-axis on the left since we're not using it for data
    ax1.set_yticks([])
    ax1.spines['left'].set_visible(False)

    # Format date axis
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Include year in format
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=12)  # Rotate for better visibility

    # Add grid for better readability
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    # Calculate the date range for dynamic title
    start_year = merged_df['date'].min().year
    end_year = merged_df['date'].max().year
    year_range = f"{start_year}" if start_year == end_year else f"{start_year}-{end_year}"

    # Add title
    plt.title(f"{market_name} Volatility and News Sentiment ({year_range})", fontsize=20, fontweight='bold', pad=20)

    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=13, frameon=True, fancybox=True)

    # Add a subtle background color for better contrast
    fig.patch.set_facecolor('#f8f9fa')

    plt.tight_layout(pad=2)
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show the plot
    if show_plot:
        plt.show()
    else:
        plt.close()


# Plot predicted vs actual volatility
def plot_prediction_results(test_dates: np.ndarray, y_test_inv: np.ndarray, y_pred_inv: np.ndarray,
                           market_name: str = 'Market',
                           save_path: Optional[str] = None,
                           show_plot: bool = True) -> None:
    """
    Plot predicted vs actual volatility.
    
    Args:
        test_dates: Array of dates for the x-axis.
        y_test_inv: Array of actual volatility values.
        y_pred_inv: Array of predicted volatility values.
        market_name: Name of the market for plot titles and labels.
        save_path: Path to save the plot image, or None to skip saving.
        show_plot: Whether to display the plot.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, y_test_inv, label='Actual Volatility', color='#2176ae', linewidth=2)
    plt.plot(test_dates, y_pred_inv, label='Predicted Volatility (LSTM)', color='#d7263d', linewidth=2, linestyle='--')

    # Format date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    plt.title(f'LSTM Volatility Prediction vs. Actual ({market_name})', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Volatility', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=12)

    # Add tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show the plot
    if show_plot:
        plt.show()
    else:
        plt.close()


# Run the entire volatility analysis pipeline
def run_volatility_pipeline(news_df: pd.DataFrame, 
                           stock_data: pd.DataFrame,
                           market_name: str,
                           cut_date: str,
                           output_dir: str = "../news",
                           seq_len: int = 10,
                           epochs: int = 50,
                           batch_size: int = 16,
                           learning_rate: float = 0.001,
                           use_sentiment: bool = True,                           verbose: bool = True) -> Dict:
    """
    Run the entire volatility analysis pipeline.
    
    Args:
        news_df: DataFrame containing news data.
        stock_data: DataFrame containing stock market data.
        market_name: Name of the market (e.g., 'FTSE 100', 'IBEX 35').
        cut_date: Date string to split training/testing data (format: 'YYYY-MM-DD').
        output_dir: Directory to save output plots.
        seq_len: Sequence length for LSTM.
        epochs: Number of epochs for LSTM training.
        batch_size: Batch size for LSTM training.
        learning_rate: Learning rate for optimizer.
        use_sentiment: Whether to use sentiment inference for predictions. If False, only uses previous volatility.
        verbose: Whether to print progress information.
    
    Returns:
        Dictionary containing model metrics and other results.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure we have the right date format
    news_df['date'] = pd.to_datetime(news_df['date'])
    news_daily = news_df.resample('D', on='date').count().reset_index()
    
    # Ensure stock data is properly formatted
    if 'Date' in stock_data.columns:
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        # Make timezone-naive if needed
        if hasattr(stock_data['Date'].dtype, 'tz') and stock_data['Date'].dt.tz is not None:
            stock_data['Date'] = stock_data['Date'].dt.tz_localize(None)
    
    # Merge data
    merged = pd.merge(news_daily, stock_data, left_on='date', right_on='Date', how='inner')
    merged['Volatility_Smooth'] = merged['Volatility'].rolling(window=5, min_periods=1, center=True).mean()
    merged.rename(columns={'title': 'count',}, inplace=True)
    
    # Clean up the merged dataframe
    if 'Date' in merged.columns:
        merged.drop(columns=['Date'], inplace=True)
    
    # Add technical indicators to stock data first
    stock_data_enhanced = add_technical_indicators(stock_data)
    
    # Re-merge with enhanced stock data to get technical indicators
    merged = pd.merge(news_daily, stock_data_enhanced, left_on='date', right_on='Date', how='inner')
    merged['Volatility_Smooth'] = merged['Volatility'].rolling(window=5, min_periods=1, center=True).mean()
    merged.rename(columns={'title': 'count'}, inplace=True)
    if 'Date' in merged.columns:
        merged.drop(columns=['Date'], inplace=True)
    
    # Conditionally initialize sentiment model and calculate sentiment    # Start with a base set of features that we know will be available
    base_features = ['Volatility_Smooth']
    tech_features = []
    sentiment_features = []
    
    # Add technical indicators that are available
    if 'RSI' in merged.columns:
        tech_features.extend(['RSI', 'MA_Ratio'])
    if 'Volume_Ratio' in merged.columns:
        tech_features.append('Volume_Ratio')
        
    if use_sentiment:
        # Initialize sentiment model
        tokenizer, model = initialize_sentiment_model()
        
        # Calculate enhanced sentiment for each date
        if verbose:
            print("Calculating enhanced sentiment scores...")
        merged['sentiment'] = merged['date'].apply(
            lambda date: enhanced_sentiment_calculation(date, news_df, tokenizer, model, verbose=False))
        
        # Add sentiment features
        merged = improve_sentiment_features(merged)
        
        # Shift sentiment features to use previous day's data
        merged['sentiment'] = merged['sentiment'].shift(1)
        merged['sentiment_3d'] = merged['sentiment_3d'].shift(1)
        
        # Add sentiment features 
        sentiment_features = ['sentiment', 'sentiment_3d', 'sentiment_vol', 'news_sentiment_interaction']
    else:
        # Skip sentiment calculation and use technical indicators
        if verbose:
            print("Skipping sentiment calculation (use_sentiment=False)...")
    
    # Combine all selected features
    feature_cols = base_features + tech_features
    
    if use_sentiment:
        feature_cols += sentiment_features
        
    # Ensure all columns exist in the dataframe
    missing_cols = [col for col in feature_cols if col not in merged.columns]
    if missing_cols:
        if verbose:
            print(f"Warning: Missing columns in merged data: {missing_cols}")
            print(f"Available columns: {merged.columns}")
        # Add missing columns with zeros
        for col in missing_cols:
            merged[col] = 0
            
    if verbose:
        print(f"Final feature columns for model: {feature_cols}")
    
    # Plot volatility and news count
    news_plot_path = os.path.join(output_dir, f"{market_name.lower().replace(' ', '_')}_news_count_plot.png")
    plot_volatility_news_count(merged, market_name, news_plot_path, show_plot=verbose)
    
    # Plot volatility and sentiment (only if using sentiment)
    if use_sentiment:
        sentiment_plot_path = os.path.join(output_dir, f"{market_name.lower().replace(' ', '_')}_sentiment_plot.png")
        plot_volatility_sentiment(merged, market_name, sentiment_plot_path, show_plot=verbose)
    
    # Split data for LSTM model
    if verbose:
        print(f"Splitting data at {cut_date}...")
    train, test, val = split_data(merged, cut_date)
    
    # Create validation split from training data
    
# Prepare LSTM data with validation set
    if verbose:
        print("Preparing data for LSTM model...")
    (X_train_seq, y_train_seq, X_test_seq, y_test_seq, X_val_seq, y_val_seq, 
     scaler_x, scaler_y, scaler_y_single) = prepare_lstm_data(train, test, val, feature_cols=feature_cols, seq_len=seq_len)
      # Prepare validation data using the training scalers to maintain consistency
    # Train LSTM model with early stopping
    if verbose:
        print(f"Training enhanced LSTM model with early stopping...")
    input_size = X_train_seq.shape[2]
    output_size = y_train_seq.shape[1]
    model = train_with_early_stopping(X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                                     input_size, output_size, epochs=epochs, 
                                     learning_rate=learning_rate, verbose=verbose)
    
    # Evaluate model
    if verbose:
        print("Evaluating enhanced LSTM model...")
    y_pred_inv, y_test_inv, metrics = evaluate_lstm_model(model, X_test_seq, y_test_seq, scaler_y)
    
    # Plot prediction results
    test_dates = test['date'].iloc[seq_len:seq_len+len(y_test_inv)].values
    pred_plot_path = os.path.join(output_dir, f"{market_name.lower().replace(' ', '_')}_prediction_plot.png")
    plot_prediction_results(test_dates, y_test_inv, y_pred_inv, market_name, pred_plot_path, show_plot=verbose)
    
    return {
        'model': model,
        'metrics': metrics,
        'train_size': len(train),
        'val_size': len(val),
        'test_size': len(test),
        'feature_cols': feature_cols,
        'y_pred': y_pred_inv,
        'y_actual': y_test_inv,
        'test_dates': test_dates
    }

# Add these functions after the existing sentiment functions

def add_technical_indicators(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators as additional features."""
    stock_data = stock_data.copy()
    
    # RSI (Relative Strength Index)
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    stock_data['RSI'] = calculate_rsi(stock_data['Close'])
    
    # Moving averages
    stock_data['MA_5'] = stock_data['Close'].rolling(window=5).mean()
    stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA_Ratio'] = stock_data['MA_5'] / stock_data['MA_20']
    
    # Volume indicators (if available)
    if 'Volume' in stock_data.columns:
        stock_data['Volume_MA'] = stock_data['Volume'].rolling(window=20).mean()
        stock_data['Volume_Ratio'] = stock_data['Volume'] / stock_data['Volume_MA']
    
    return stock_data

def improve_sentiment_features(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Add improved sentiment features."""
    merged_df = merged_df.copy()
    
    # Sentiment momentum (3-day rolling average)
    merged_df['sentiment_3d'] = merged_df['sentiment'].rolling(window=3, min_periods=1).mean()
    
    # Sentiment volatility (rolling std)
    merged_df['sentiment_vol'] = merged_df['sentiment'].rolling(window=5, min_periods=1).std()
    
    # News volume impact
    merged_df['news_sentiment_interaction'] = merged_df['count'] * merged_df['sentiment']
    
    return merged_df

def enhanced_sentiment_calculation(date: datetime, news_df: pd.DataFrame, tokenizer: Any, model: Any, verbose: bool = False) -> Optional[float]:
    """Enhanced sentiment with confidence weighting."""
    titles = get_titles(date, news_df)
    if not titles:
        return None
    
    sentiments = []
    confidences = []
    
    for title in titles:
        inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        
        # Get confidence (max probability)
        confidence = torch.max(probs).item()
        sentiment = probs.argmax(dim=1).item()
        
        # Map sentiment values (0, 1, 2) to (-1, 0, 1)
        if sentiment == 2:
            sentiment = -1
            
        sentiments.append(sentiment)
        confidences.append(confidence)
    
    # Weighted average by confidence
    if confidences:
        weighted_sentiment = np.average(sentiments, weights=confidences)
        return weighted_sentiment
    
    return np.mean(sentiments)


def train_with_early_stopping(X_train_seq: torch.Tensor, y_train_seq: torch.Tensor, 
                              X_val_seq: torch.Tensor, y_val_seq: torch.Tensor,
                              input_size: int, output_size: int, 
                              epochs: int = 100, patience: int = 10, 
                              learning_rate: float = 0.001, verbose: bool = True) -> nn.Module:
    """
    Train model with early stopping and learning rate scheduling.
    
    Args:
        X_train_seq: Input sequences for training.
        y_train_seq: Target values for training.
        X_val_seq: Input sequences for validation.
        y_val_seq: Target values for validation.
        input_size: Number of input features.
        output_size: Number of output values.
        epochs: Maximum number of training epochs.
        patience: Number of epochs to wait for improvement before stopping.
        learning_rate: Learning rate for optimizer.
        verbose: Whether to print training progress.
    
    Returns:
        Trained LSTM model.
    """
    model = ImprovedLSTMVolatility(input_size, output_size=output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for i in range(0, len(X_train_seq), 32):  # Batch processing
            batch_x = X_train_seq[i:i+32]
            batch_y = y_train_seq[i:i+32]
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_seq)
            val_loss = criterion(val_output, y_val_seq).item()
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        if verbose and ((epoch+1) % 10 == 0 or epoch == 0):
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(X_train_seq)*32:.5f}, Val Loss: {val_loss:.5f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model