import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SENTIMENT ANALYSIS & THEME EXTRACTION")
print("=" * 70 + "\n")


print("Loading cleaned data...")
df = pd.read_csv('/kaggle/input/cleaned-final/02_cleaned_comments_final.csv')  #Edit the Input file path
print(f"Starting with: {len(df)} comments\n")

print("=" * 70)
print("STEP 1: SENTIMENT ANALYSIS")
print("=" * 70 + "\n")

print("Analyzing sentiment...")

def get_sentiment(text):
    """Classify sentiment as Positive, Negative, or Neutral"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # -1 to 1
    
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'


df['sentiment'] = df['text_clean'].apply(get_sentiment)
df['polarity_score'] = df['text_clean'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity_score'] = df['text_clean'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

print("  Sentiment analysis complete\n")


print("Sentiment Distribution:")
print(df['sentiment'].value_counts())
print(f"\nSentiment Percentages:")
print(df['sentiment'].value_counts(normalize=True) * 100)
print(f"\nAverage polarity score: {df['polarity_score'].mean():.3f}")
print(f"Average subjectivity score: {df['subjectivity_score'].mean():.3f}\n")


print("=" * 70)
print("STEP 2: THEME EXTRACTION")
print("=" * 70 + "\n")

print("Extracting themes using TF-IDF + K-Means...")


vectorizer = TfidfVectorizer(
    max_features=100,
    stop_words='english',
    min_df=2,
    max_df=0.8
)
X = vectorizer.fit_transform(df['text_clean'])


n_themes = 6
kmeans = KMeans(n_clusters=n_themes, random_state=42, n_init=10)
df['theme'] = kmeans.fit_predict(X)


terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

print(f"\n  Extracted {n_themes} themes:\n")

themes = {}
for i in range(n_themes):
    top_words = [terms[ind] for ind in order_centroids[i, :8]]
    theme_name = ' / '.join(top_words[:5])
    themes[i] = theme_name
    print(f"Theme {i}: {theme_name}")

df['theme_name'] = df['theme'].map(themes)

print(f"\n  Theme extraction complete\n")


df.to_csv('03_analyzed_comments.csv', index=False)
print(f"  Saved analyzed data to: data/03_analyzed_comments.csv\n")


print("=" * 70)
print("ANALYSIS STATISTICS")
print("=" * 70 + "\n")

print("Sentiment by Query (top 5 queries):")
top_queries = df['query'].value_counts().head(5).index
for query in top_queries:
    query_df = df[df['query'] == query]
    print(f"\n{query}:")
    print(f"  Count: {len(query_df)}")
    print(f"  Sentiment: {query_df['sentiment'].value_counts().to_dict()}")
    print(f"  Avg polarity: {query_df['polarity_score'].mean():.3f}")

print("\n" + "=" * 70)
print("Sentiment by Theme:")
print("=" * 70)
sentiment_by_theme = pd.crosstab(df['theme_name'], df['sentiment'])
print(sentiment_by_theme)

print("\n  Analysis complete! Ready for visualization.\n")


