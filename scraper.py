import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
import html
import re
import os

# HackerNews Algolia API (free, no auth)
base_url = "https://hn.algolia.com/api/v1/search"

queries = [
    "AI workplace",
    "artificial intelligence job",
    "ChatGPT work",
    "adapting to AI workplace",
    "AI career development",
    "Gen Z AI workplace",
    "will AI replace my job",
    "AI threat to employment",
    "AI restructuring work roles",
    "adapting to artificial intelligence workplace",
    "automation job security",
    "job security AI",
    "AI skills learning",
    "future of work",
    "AI adoption employee",
    "generative AI work",
    "AI change nature of work",
    "job redesign artificial intelligence",
    "how AI transforms tasks",
    "career resilience artificial intelligence",
    "worker adaptation technology change",
    "AI skills gap workforce",
    "upskilling for future work",
    "what skills does AI require",
    "training AI adoption employee",
    "trust AI workplace decisions",
    "algorithmic bias hiring",
    "fairness AI systems work",
    "transparency AI decision making",
    "older workers AI displacement",
    "experienced professionals AI",
    "knowledge workers AI impact",
    "AI job displacement fear",
    "employment impact AI",
    "AI work life balance",
    "employee wellbeing AI adoption",
    "work satisfaction AI",
    "company strategy AI implementation",
    "change management AI workplace",
    "organizational support AI workers",
    "human AI collaboration work",
    "working alongside AI",
    "AI coworker productivity",
    "career prospects AI era",
    "career commitment automation",
    "job satisfaction AI introduction"
]

all_comments = []

print("=" * 70)
print("SCRAPING HACKERNEWS FOR AI & WORK DISCUSSIONS")
print("=" * 70 + "\n")

for query in queries:
    print(f"  Searching: '{query}'...")
    
    params = {
        'query': query,
        'tags': 'comment',
        'hitsPerPage': 100,
        'numericFilters': 'created_at_i>1700000000'  
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()
        
        hits = data.get('hits', [])
        print(f"     Found {len(hits)} results")
        
        for hit in hits:
            comment_text = hit.get('comment_text', '')
            
            # Skip short comments, I think they wont give a complete picture so as to say
            if not comment_text or len(comment_text) < 50:
                continue
            if comment_text in ['[deleted]', '[removed]', 'null']:
                continue
            
            all_comments.append({
                'source': 'HackerNews',
                'query': query,
                'text': comment_text,
                'author': hit.get('author', 'unknown'),
                'created_at': hit.get('created_at', ''),
                'points': hit.get('points', 0),
                'url': f"https://news.ycombinator.com/item?id={hit['objectID']}"
            })
        
        time.sleep(1)  
    
    except Exception as e:
        print(f"     Error: {str(e)}")

print("\n" + "=" * 70)
print("SCRAPING COMPLETE")
print(f"Total comments collected: {len(all_comments)}")
print("=" * 70)


df = pd.DataFrame(all_comments)

if len(df) == 0:
    print("  No data collected. Check your internet connection.")
    exit()

# Remove exact duplicates
df_dedup = df.drop_duplicates(subset=['text'])
print(f"  After removing duplicates: {len(df_dedup)} comments")

# Sorting 
df_dedup = df_dedup.sort_values('created_at', ascending=False)


os.makedirs('data', exist_ok=True)
output_file = 'data/01_raw_hackernews_comments.csv'
df_dedup.to_csv(output_file, index=False)

print(f"  Saved to: {output_file}\n")


print("=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(f"\nComments by search query:")
print(df_dedup['query'].value_counts())
print(f"\nAverage engagement (points): {df_dedup['points'].mean():.1f}")
print(f"Date range: {df_dedup['created_at'].min()} to {df_dedup['created_at'].max()}")


print("\n" + "=" * 70)
print("SAMPLE COMMENTS")
print("=" * 70)
for i, row in df_dedup.head(3).iterrows():
    print(f"\n[Sample {i+1}] Query: {row['query']} | Score: {row['points']}")
    print(f"{row['text'][:200]}...")
    print("-" * 70)

print("\n  Scraping complete! Ready for preprocessing.")
