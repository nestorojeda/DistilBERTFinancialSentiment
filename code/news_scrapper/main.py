#!/usr/bin/env python3
"""
Financial News Scraper using News API
Fetches financial news in Spanish, French, and German by sentiment categories
Each sentiment category (Positive, Neutral, Negative) is processed separately
"""

import os
import json
import datetime
import requests
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Get the API key
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Get the base path for storing the news
BASE_PATH = os.getenv("BASE_PATH")
if not BASE_PATH:
    raise ValueError("BASE_PATH not found in .env file")

if not NEWS_API_KEY:
    raise ValueError("NEWS_API_KEY not found in .env file")

# Base URL for News API
BASE_URL = "https://newsapi.org/v2/everything"

# Languages to fetch
LANGUAGES = {
    "es": "Spanish",
    "fr": "French",
    "de": "German"
}

# Sentiment categories
SENTIMENTS = ["positive", "neutral", "negative"]

def load_keywords_by_sentiment():
    """
    Load keywords from the keywords.csv file organized by language and sentiment
    
    Returns:
        dict: Dictionary with language codes as keys and dictionaries of sentiment categories and keywords as values
    """
    # Get the path to the keywords.csv file
    keywords_path = Path(__file__).parent / "keywords.csv"
    
    # Read the CSV file
    keywords_df = pd.read_csv(keywords_path)
    
    # Initialize the keywords dictionary
    keywords = {lang: {"positive": [], "negative": [], "neutral": []} for lang in LANGUAGES.keys()}
    
    # Map language codes to column names in the CSV
    lang_columns = {
        "es": "Spanish",
        "fr": "French",
        "de": "German"
    }
    
    # Populate the keywords dictionary
    for _, row in keywords_df.iterrows():
        category = row["Category"]
        for lang_code, column_name in lang_columns.items():
            keyword = row[column_name]
            if pd.notna(keyword) and keyword.strip():  # Check for non-empty values
                keywords[lang_code][category].append(keyword.strip())
    
    return keywords

def get_search_query_for_sentiment(language, keywords, sentiment):
    """
    Create a search query string for the News API using keywords for a specific sentiment
    
    Args:
        language (str): Language code ('es', 'fr', 'de')
        keywords (dict): Dictionary containing keywords for each language and sentiment
        sentiment (str): Sentiment category ("positive", "neutral", "negative")
        
    Returns:
        str: Search query string
    """
    # Use only keywords for the specified sentiment and language
    sentiment_keywords = keywords[language][sentiment]
    
    # Create a query with OR operators
    if sentiment_keywords:
        query = " OR ".join(sentiment_keywords)
        return query
    else:
        return ""

def fetch_financial_news_by_sentiment(language, keywords, sentiment, num_articles=10):
    """
    Fetch financial news articles for a specific language and sentiment category
    
    Args:
        language (str): Language code ('es', 'fr', 'de')
        keywords (dict): Dictionary containing keywords for each language and sentiment
        sentiment (str): Sentiment category ("positive", "neutral", "negative")
        num_articles (int): Number of articles to fetch
        
    Returns:
        list: List of articles with attached sentiment and matching keyword information
    """
    query = get_search_query_for_sentiment(language, keywords, sentiment)
    if not query:
        print(f"No {sentiment} keywords available for {LANGUAGES[language]}")
        return []
    
    params = {
        'q': query,
        'language': language,
        'sortBy': 'publishedAt',
        'pageSize': num_articles,
        'apiKey': NEWS_API_KEY,
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        articles = response.json()['articles']
        
        # Attach the sentiment and matching keyword to each article
        for article in articles:
            article['sentiment'] = sentiment
            
            # Find which keyword from this sentiment category appears in the title
            title = article['title'].lower()
            matching_keywords = [kw for kw in keywords[language][sentiment] 
                               if kw.lower() in title]
            article['matching_keyword'] = matching_keywords[0] if matching_keywords else ""
            
        return articles
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {sentiment} news for {LANGUAGES[language]}: {e}")
        return []

def save_news_by_sentiment_to_csv(all_articles, language):
    """
    Save news articles organized by sentiment category to a CSV file
    The CSV will contain the headline, keyword with sentiment, and publish date
    
    Args:
        all_articles (list): List of news articles with sentiment information
        language (str): Language code ('es', 'fr', 'de')
    """
    if not all_articles:
        print(f"No articles to save for {LANGUAGES[language]}")
        return
    
    # Format the date
    today = datetime.datetime.now().strftime("%Y%m%d")
    
    # Create output directory if it doesn't exist
    output_dir = Path(BASE_PATH + "/data/news")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a DataFrame with only headline, keyword sentiment, and publish date
    result_data = []
    for article in all_articles:
        # Include all articles regardless of whether there's a matching keyword
        result_data.append({
            'headline': article['title'],
            'sentiment': article['sentiment'],
            'publish_date': article['publishedAt']
        })
    
    # Create a DataFrame from the results
    result_df = pd.DataFrame(result_data)
    
    # Save to CSV
    output_file = output_dir / f"news_{language}_{today}.csv"
    
    if not result_df.empty:
        result_df.to_csv(output_file, index=False)
        print(f"Saved {len(result_data)} {LANGUAGES[language]} headlines to {output_file}")
    else:
        print(f"No headlines found for {LANGUAGES[language]}")

def main():
    """
    Main function to fetch financial news by sentiment categories for different languages
    and save them with their associated sentiment information
    """
    print("Starting to fetch financial news by sentiment categories...")
    
    # Load keywords organized by sentiment
    keywords = load_keywords_by_sentiment()
    
    # Process each language
    for lang_code, lang_name in LANGUAGES.items():
        print(f"\nProcessing news in {lang_name}...")
        
        all_articles_with_sentiment = []
        
        # Process each sentiment category separately
        for sentiment in SENTIMENTS:
            print(f"  Fetching {sentiment.lower()} news in {lang_name}...")
            articles = fetch_financial_news_by_sentiment(lang_code, keywords, sentiment)
            
            if articles:
                print(f"  Found {len(articles)} {sentiment.lower()} news articles in {lang_name}")
                all_articles_with_sentiment.extend(articles)
            else:
                print(f"  No {sentiment.lower()} news found for {lang_name}")
        
        # Save results for this language
        if all_articles_with_sentiment:
            print(f"Saving news articles with sentiment for {lang_name}...")
            save_news_by_sentiment_to_csv(all_articles_with_sentiment, lang_code)
        else:
            print(f"No news found for {lang_name}")
    
    print("\nNews fetching and categorization by sentiment completed!")

if __name__ == "__main__":
    main()