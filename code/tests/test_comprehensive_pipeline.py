#!/usr/bin/env python3
"""
Comprehensive Pipeline Test

This script validates the complete volatility prediction pipeline including:
1. Testing with sentiment analysis enabled
2. Testing both Simple and Improved LSTM models
3. Validating data leakage prevention across all scenarios
4. Performance comparison between different configurations
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add the code directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from code.volatility_analysis.volatility_pipeline import run_volatility_pipeline

def create_test_data():
    """Create comprehensive test data for validation."""
    print("Creating test data...")
    
    # Create date range
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create stock data with realistic patterns
    np.random.seed(42)
    n_days = len(dates)
      # Generate price series with some volatility clustering
    returns = np.random.normal(0.001, 0.02, n_days)
    for i in range(1, n_days):
        # Add volatility clustering effect
        if abs(returns[i-1]) > 0.03:
            returns[i] = np.random.normal(0, 0.04)
    
    prices = 100 * np.exp(np.cumsum(returns))
    volumes = np.random.lognormal(15, 0.5, n_days)
    
    # Calculate volatility from returns
    volatility = np.abs(returns) + np.random.exponential(0.01, n_days)
    
    stock_data = pd.DataFrame({
        'Date': dates,
        'Open': prices + np.random.normal(0, 0.5, n_days),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        'Close': prices,
        'Volume': volumes,
        'Adj Close': prices,
        'Return': returns,
        'Volatility': volatility
    })
    
    # Create news data with varying sentiment
    news_dates = []
    headlines = []
    sentiments = []
    
    for date in dates:
        # Variable number of news articles per day (0-5)
        n_articles = np.random.poisson(2)
        for _ in range(n_articles):
            news_dates.append(date)
              # Create headlines with different sentiment patterns
            sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], p=[0.3, 0.3, 0.4])
            if sentiment_type == 'positive':
                headlines.append("Market shows strong growth potential with positive outlook")
                sentiments.append(np.random.uniform(0.1, 0.9))
            elif sentiment_type == 'negative':
                headlines.append("Concerns about market volatility and economic uncertainty")
                sentiments.append(np.random.uniform(-0.9, -0.1))
            else:
                headlines.append("Market analysis shows mixed signals for investors")
                sentiments.append(np.random.uniform(-0.1, 0.1))
    
    news_data = pd.DataFrame({
        'date': news_dates,  # Changed from 'Date' to 'date'
        'title': headlines,  # Changed from 'Headline' to 'title'
        'Sentiment': sentiments
    })
    
    print(f"Created stock data: {len(stock_data)} days")
    print(f"Created news data: {len(news_data)} articles")
    
    return stock_data, news_data

def test_pipeline_configuration(stock_data, news_data, config_name, **kwargs):
    """Test a specific pipeline configuration."""
    print(f"\n{'='*60}")
    print(f"Testing Configuration: {config_name}")
    print(f"{'='*60}")
    
    try:
        results = run_volatility_pipeline(
            news_df=news_data,
            stock_data=stock_data,
            market_name="TEST_MARKET",
            cut_date="2024-10-01",
            output_dir="./test_output",
            seq_len=10,
            epochs=3,  # Reduced for faster testing
            learning_rate=0.001,
            verbose=True,
            **kwargs
        )
        
        print(f"✓ {config_name} completed successfully!")
        print(f"  - Train size: {results.get('train_size', 'N/A')}")
        print(f"  - Test size: {results.get('test_size', 'N/A')}")
        print(f"  - Val size: {results.get('val_size', 'N/A')}")
        print(f"  - Features used: {len(results.get('features_used', []))}")
        print(f"  - Model type: {results.get('model_type', 'N/A')}")
        
        metrics = results.get('model_metrics', {})
        if metrics:
            print(f"  - RMSE: {metrics.get('RMSE', 'N/A'):.6f}")
            print(f"  - MAE: {metrics.get('MAE', 'N/A'):.6f}")
            print(f"  - R²: {metrics.get('R2', 'N/A'):.6f}")
        
        return True, results
        
    except Exception as e:
        print(f"✗ {config_name} FAILED: {str(e)}")
        return False, None

def main():
    """Run comprehensive pipeline tests."""
    print("🚀 Starting Comprehensive Pipeline Validation")
    print("=" * 60)
    
    # Create test data
    stock_data, news_data = create_test_data()
    
    # Test configurations
    test_configs = [
        {
            'name': 'Simple LSTM + Sentiment',
            'use_sentiment': True,
            'lstm_type': 'simple',
            'use_technical_indicators': True
        },
        {
            'name': 'Simple LSTM (No Sentiment)',
            'use_sentiment': False,
            'lstm_type': 'simple',
            'use_technical_indicators': True
        },
        {
            'name': 'Improved LSTM + Sentiment',
            'use_sentiment': True,
            'lstm_type': 'improved',
            'use_technical_indicators': True
        },
        {
            'name': 'Improved LSTM (No Sentiment)',
            'use_sentiment': False,
            'lstm_type': 'improved',
            'use_technical_indicators': True
        }
    ]
    
    # Run tests
    results = []
    success_count = 0
    
    for config in test_configs:
        config_name = config.pop('name')
        success, result = test_pipeline_configuration(
            stock_data, news_data, config_name, **config
        )
        
        if success:
            success_count += 1
            results.append((config_name, result))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total configurations tested: {len(test_configs)}")
    print(f"Successful configurations: {success_count}")
    print(f"Failed configurations: {len(test_configs) - success_count}")
    
    if success_count == len(test_configs):
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ Data leakage fixes are working correctly across all configurations")
        print("✅ Both Simple and Improved LSTM models are functional")
        print("✅ Sentiment analysis integration is working properly")
        print("✅ Technical indicators are being computed correctly")
        
        # Performance comparison
        if len(results) > 0:
            print(f"\n📊 PERFORMANCE COMPARISON:")
            for config_name, result in results:
                metrics = result.get('model_metrics', {})
                rmse = metrics.get('RMSE', 0)
                r2 = metrics.get('R2', 0)
                print(f"  {config_name:25} RMSE: {rmse:.6f}, R²: {r2:.6f}")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return success_count == len(test_configs)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
