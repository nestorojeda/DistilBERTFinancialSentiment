#!/usr/bin/env python3
"""
Financial News Scraper using News API
Fetches the latest financial news in Spanish, French, and German
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

# Financial keywords for each language
KEYWORDS = {
    "es": "finanzas OR economía OR mercado OR inversiones OR bolsa",
    "fr": "finance OR économie OR marché OR investissements OR bourse",
    "de": "finanzen OR wirtschaft OR markt OR investitionen OR börse"
}

def fetch_financial_news(language, num_articles=10):
    """
    Fetch financial news articles for a specific language
    
    Args:
        language (str): Language code ('es', 'fr', 'de')
        num_articles (int): Number of articles to fetch
        
    Returns:
        list: List of articles
    """
    params = {
        'q': KEYWORDS[language],
        'language': language,
        'sortBy': 'publishedAt',
        'pageSize': num_articles,
        'apiKey': NEWS_API_KEY,
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        return response.json()['articles']
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news for {LANGUAGES[language]}: {e}")
        return []

def save_to_csv(articles, language):
    """
    Save articles to a CSV file
    
    Args:
        articles (list): List of news articles
        language (str): Language code
    """
    if not articles:
        print(f"No articles to save for {LANGUAGES[language]}")
        return
    
    # Create a DataFrame from the articles
    df = pd.DataFrame(articles)
    
    # Extract the source name
    df['source'] = df['source'].apply(lambda x: x['name'] if isinstance(x, dict) and 'name' in x else '')
    
    # Format the date
    today = datetime.datetime.now().strftime("%Y%m%d")
    
    # Create directory if it doesn't exist
    output_dir = Path("../../data/news")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save to CSV
    output_file = output_dir / f"financial_news_{language}_{today}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved {len(articles)} {LANGUAGES[language]} articles to {output_file}")

def main():
    """
    Main function to fetch and save financial news in different languages
    """
    print("Starting to fetch financial news...")
    
    for lang_code, lang_name in LANGUAGES.items():
        print(f"Fetching news in {lang_name}...")
        articles = fetch_financial_news(lang_code)
        save_to_csv(articles, lang_code)
    
    print("News fetching completed!")

if __name__ == "__main__":
    main()