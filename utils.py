def transform_labels(example):
    # Create a mapping dictionary for sentiment labels
    sentiment_map = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }
    
    # Get the sentiment value using the map with a default
    # This handles both cases in one step and makes the code more maintainable
    return {'labels': sentiment_map.get(example['Sentiment'].lower(), 0)}

