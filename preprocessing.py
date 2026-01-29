import pandas as pd
import re
import html
import os

print("=" * 70)
print("LOADING & CLEANING DATA")
print("=" * 70 + "\n")

print("Loading raw data...")
df = pd.read_csv('/kaggle/working/data/01_raw_hackernews_comments.csv') ##Edit the file path
print(f"Starting with: {len(df)} comments\n")


df = df.dropna(subset=['text'])
print(f"  After removing NaN: {len(df)} comments")


df = df[df['text'].str.len() >= 40]
print(f"  After removing short comments: {len(df)} comments")


def clean_text(text):
    """
    Comprehensive text cleaning for HackerNews comments
    """
    
    # STEP 1: Decode HTML entities
    text = html.unescape(text)
    
    # STEP 2: Fix URL-encoded characters
    text = text.replace('x2F', '/')
    text = text.replace('x27', "'")
    text = text.replace('x3A', ':')
    text = text.replace('x3D', '=')
    text = text.replace('x26', '&')
    text = text.replace('x25', '%')
    text = text.replace('x2D', '-')
    text = text.replace('x5F', '_')
    
    # STEP 3: Fix partial HTML tags
    text = text.replace('gt', '>')
    text = text.replace('lt', '<')
    text = text.replace('quot', '"')
    text = text.replace('iquot', '"')
    text = text.replace('amp', '&')
    
    # STEP 4: Handle 'p' tags (paragraph markers)
    # Only replace 'p' when it's isolated (word boundary)
    text = re.sub(r'\bp\s+', ' ', text)      # p followed by space
    text = re.sub(r'\s+p\b', ' ', text)      # space followed by p
    text = re.sub(r'^p\s', '', text)         # p at start
    text = re.sub(r'\sp$', '', text)         # p at end
    
    # STEP 5: Remove or fix italic/bold markers
    text = re.sub(r'</?[ib]>', '', text)     # Remove <i>, </i>, <b>, </b>
    text = re.sub(r'\bi\s+', '', text)       # Remove standalone 'i ' 
    text = re.sub(r'\bb\s+', '', text)       # Remove standalone 'b '
    
    # STEP 6: Remove remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # STEP 7: Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'href[^>]*>', '', text)
    
    # STEP 8: Fix common formatting issues
    text = re.sub(r'quot', '"', text)        # lingering quotes
    text = re.sub(r'apos', "'", text)        # apostrophes
    
    # STEP 9: Clean up multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = text.strip()
    
    return text

print("\nCleaning text (enhanced)...\n")
df['text_clean'] = df['text'].apply(clean_text)


df = df.drop_duplicates(subset=['text_clean'])
print(f"  After removing duplicates: {len(df)} comments")


df = df[df['text_clean'].str.len() > 0]
print(f"  After removing empty: {len(df)} comments\n")


os.makedirs('data', exist_ok=True)
output_file = 'data/02_cleaned_comments_final.csv'  ##specify the O/P name
df.to_csv(output_file, index=False)
print(f"  Saved to: {output_file}")


print("\n" + "=" * 70)
print("SAMPLE COMMENTS (BEFORE & AFTER CLEANING)")
print("=" * 70)

for i, row in df.head(5).iterrows():
    print(f"\n[Sample {i+1}] Query: {row['query']}")
    print(f"\n  RAW (first 150 chars):")
    print(f"  {row['text'][:150]}...")
    print(f"\n  CLEANED (first 150 chars):")
    print(f"  {row['text_clean'][:150]}...")
    print("\n" + "-" * 70)

# ============================================================================
# FINAL Count of data
# ============================================================================
print("\n" + "=" * 70)
print("FINAL DATA Stat")
print("=" * 70)
print(f"\nTotal comments ready for analysis: {len(df)}")
print(f"Average comment length: {df['text_clean'].str.len().mean():.0f} characters")
print(f"Min/Max length: {df['text_clean'].str.len().min()} / {df['text_clean'].str.len().max()}")
print(f"\nComments by query (top 10):")
print(df['query'].value_counts().head(10))

print("\n" + "=" * 70)
print("  PREPROCESSING COMPLETE!")
print("=" * 70)
print("\nYour cleaned data is ready for sentiment analysis!")
print(f"File: {output_file}")

