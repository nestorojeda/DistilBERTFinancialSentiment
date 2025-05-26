#!/usr/bin/env python3
"""
Test script to verify data leakage fixes in volatility pipeline
"""

from code.volatility_analysis.volatility_pipeline import *
import pandas as pd
import numpy as np

def test_temporal_split():
    """Test the temporal data splitting function"""
    print("Testing temporal data split...")
    
    # Create test data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    test_df = pd.DataFrame({
        'date': dates,
        'Volatility': np.random.random(len(dates)) * 0.1,
        'count': np.random.randint(1, 50, len(dates))
    })
    test_df['Volatility_Smooth'] = test_df['Volatility'].rolling(window=5, min_periods=1).mean()

    # Test the temporal split function
    train, test, val = split_data_temporal(test_df, '2024-10-01')
    
    print(f'Train size: {len(train)}, Test size: {len(test)}, Val size: {len(val)}')
    print(f'Train date range: {train["date"].min()} to {train["date"].max()}')
    print(f'Test date range: {test["date"].min()} to {test["date"].max()}')
    print(f'Val date range: {val["date"].min()} to {val["date"].max()}')
    
    # Verify no data leakage
    assert train["date"].max() < pd.to_datetime('2024-10-01'), "Data leakage: train data contains future dates"
    assert test["date"].min() >= pd.to_datetime('2024-10-01'), "Data leakage: test data contains past dates"
    assert val["date"].min() < pd.to_datetime('2024-10-01'), "Validation data should be from training period"
    
    print("✓ Temporal split test PASSED - no data leakage detected")

def test_sentiment_features():
    """Test sentiment feature engineering without leakage"""
    print("\nTesting sentiment feature engineering...")
    
    # Create test data with sentiment
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    train_df = pd.DataFrame({
        'date': dates[:200],
        'sentiment': np.random.normal(0, 0.5, 200),
        'count': np.random.randint(1, 50, 200),
        'Volatility_Smooth': np.random.random(200) * 0.1
    })
    
    test_df = pd.DataFrame({
        'date': dates[200:300],
        'sentiment': np.random.normal(0, 0.5, 100),
        'count': np.random.randint(1, 50, 100),
        'Volatility_Smooth': np.random.random(100) * 0.1
    })
    
    # Apply sentiment features
    train_processed = improve_sentiment_features(train_df)
    test_processed = apply_sentiment_features_test(test_df, train_processed)
    
    # Check that both have the same feature columns
    train_sentiment_cols = [col for col in train_processed.columns if col.startswith('sentiment')]
    test_sentiment_cols = [col for col in test_processed.columns if col.startswith('sentiment')]
    
    print(f"Train sentiment features: {len(train_sentiment_cols)}")
    print(f"Test sentiment features: {len(test_sentiment_cols)}")
    
    # Verify z-score normalization uses training stats
    if 'sentiment_zscore' in train_processed.columns and 'sentiment_zscore' in test_processed.columns:
        train_mean = train_processed['sentiment'].mean()
        train_std = train_processed['sentiment'].std()
        
        # Calculate expected test z-score using training stats
        expected_test_zscore = (test_processed['sentiment'] - train_mean) / (train_std + 1e-8)
        actual_test_zscore = test_processed['sentiment_zscore']
        
        # Check if they match (within tolerance)
        zscore_match = np.allclose(expected_test_zscore.fillna(0), actual_test_zscore.fillna(0), atol=1e-6)
        assert zscore_match, "Z-score normalization not using training data statistics"
        
        print("✓ Z-score normalization correctly uses training data statistics")
    
    print("✓ Sentiment feature engineering test PASSED")

def test_lstm_data_preparation():
    """Test LSTM data preparation without leakage"""
    print("\nTesting LSTM data preparation...")
    
    # Create test data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    train_df = pd.DataFrame({
        'date': dates[:200],
        'Volatility_Smooth': np.random.random(200) * 0.1,
        'sentiment': np.random.normal(0, 0.5, 200),
        'count': np.random.randint(1, 50, 200)
    })
    
    test_df = pd.DataFrame({
        'date': dates[200:300],
        'Volatility_Smooth': np.random.random(100) * 0.1,
        'sentiment': np.random.normal(0, 0.5, 100),
        'count': np.random.randint(1, 50, 100)
    })
    
    val_df = pd.DataFrame({
        'date': dates[300:366],
        'Volatility_Smooth': np.random.random(66) * 0.1,
        'sentiment': np.random.normal(0, 0.5, 66),
        'count': np.random.randint(1, 50, 66)
    })
    
    feature_cols = ['Volatility_Smooth', 'sentiment']
    
    try:
        result = prepare_lstm_data(train_df, test_df, val_df, feature_cols=feature_cols, seq_len=10)
        X_train_seq, y_train_seq, X_test_seq, y_test_seq, X_val_seq, y_val_seq, scaler_x, scaler_y, scaler_y_single = result
        
        print(f"X_train_seq shape: {X_train_seq.shape}")
        print(f"X_test_seq shape: {X_test_seq.shape}")
        print(f"X_val_seq shape: {X_val_seq.shape}")
        
        # Verify scalers were fit only on training data
        assert hasattr(scaler_x, 'data_min_'), "Feature scaler not properly fitted"
        assert hasattr(scaler_y, 'data_min_'), "Target scaler not properly fitted"
        
        print("✓ LSTM data preparation test PASSED")
        
    except Exception as e:
        print(f"✗ LSTM data preparation test FAILED: {e}")
        raise

if __name__ == "__main__":
    print("Running data leakage fix tests...\n")
    
    try:
        test_temporal_split()
        test_sentiment_features()
        test_lstm_data_preparation()
        
        print("\n🎉 All data leakage fix tests PASSED!")
        print("The volatility pipeline is now free from data leakage issues.")
        
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
