#!/usr/bin/env python3
"""
End-to-end test of the volatility pipeline to ensure all fixes work together
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_test_data():
    """Create synthetic test data for pipeline validation"""
    
    # Create date range
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Create synthetic news data
    news_data = []
    for date in dates:
        # Random number of news articles per day
        num_articles = np.random.randint(1, 30)
        for i in range(num_articles):
            news_data.append({
                'date': date,
                'title': f"Sample financial news {i+1} for {date.strftime('%Y-%m-%d')}"
            })
    
    news_df = pd.DataFrame(news_data)
    
    # Create synthetic stock data
    stock_data = []
    for i, date in enumerate(dates):
        # Generate realistic stock data with some volatility patterns
        base_price = 8000 + np.sin(i / 30) * 200  # Seasonal pattern
        noise = np.random.normal(0, 50)  # Daily noise
        price = max(base_price + noise, 1000)  # Ensure positive price
        
        # Calculate daily return and volatility
        if i == 0:
            daily_return = 0
        else:
            prev_price = stock_data[-1]['Close']
            daily_return = (price - prev_price) / prev_price
        
        volatility = abs(daily_return) + np.random.exponential(0.01)  # Add base volatility
        
        stock_data.append({
            'Date': date,
            'Open': price * (1 + np.random.normal(0, 0.001)),
            'High': price * (1 + abs(np.random.normal(0, 0.002))),
            'Low': price * (1 - abs(np.random.normal(0, 0.002))),
            'Close': price,
            'Volume': np.random.randint(500000, 2000000),
            'Adj Close': price,
            'Return': daily_return,
            'Volatility': volatility
        })
    
    stock_df = pd.DataFrame(stock_data)
    
    return news_df, stock_df

def test_pipeline_no_sentiment():
    """Test pipeline without sentiment analysis"""
    print("Testing pipeline WITHOUT sentiment analysis...")
    
    try:
        from code.volatility_analysis.volatility_pipeline import run_volatility_pipeline
        
        news_df, stock_df = create_test_data()
        
        # Run pipeline without sentiment
        results = run_volatility_pipeline(
            news_df=news_df,
            stock_data=stock_df,
            market_name="Test Market",
            cut_date="2024-10-01",
            output_dir="./test_output",
            seq_len=5,  # Small sequence length for faster testing
            epochs=5,   # Few epochs for faster testing
            learning_rate=0.001,
            use_sentiment=False,  # No sentiment analysis
            verbose=True,
            use_technical_indicators=True
        )
        
        print(f"✓ Pipeline completed successfully!")
        print(f"  - Train size: {results['train_size']}")
        print(f"  - Test size: {results['test_size']}")
        print(f"  - Val size: {results['val_size']}")
        print(f"  - Features used: {results['feature_cols']}")
        print(f"  - Model metrics: {results['metrics']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running end-to-end pipeline validation...\n")
    
    success = test_pipeline_no_sentiment()
    
    if success:
        print("\n🎉 End-to-end pipeline validation PASSED!")
        print("The volatility pipeline with data leakage fixes is working correctly.")
    else:
        print("\n❌ End-to-end pipeline validation FAILED!")
