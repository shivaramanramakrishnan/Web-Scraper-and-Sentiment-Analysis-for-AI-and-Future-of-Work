import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import numpy as np

print("=" * 70)
print("CREATING WORKER-FRIENDLY VISUALIZATIONS")
print("=" * 70 + "\n")

print("Loading analyzed data...")
df = pd.read_csv('/kaggle/working/03_analyzed_comments.csv')
print(f"Working with: {len(df)} comments\n")


os.makedirs('outputs', exist_ok=True)


plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


print("Creating Visualization 1: What Workers Think About AI...")

fig, ax = plt.subplots(figsize=(10, 8))

sentiment_counts = df['sentiment'].value_counts()
colors_map = {'Positive': '#27ae60', 'Negative': '#e74c3c', 'Neutral': '#bdc3c7'}
colors = [colors_map.get(s, '#95a5a6') for s in sentiment_counts.index]

wedges, texts, autotexts = ax.pie(
    sentiment_counts, 
    labels=sentiment_counts.index, 
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    textprops={'fontsize': 14, 'weight': 'bold'},
    explode=(0.05, 0.05, 0.05)
)


for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(16)
    autotext.set_weight('bold')

ax.set_title('WHAT DO WORKERS THINK ABOUT AI AT WORK?\n(Based on 350+ real comments)', 
             fontsize=16, fontweight='bold', pad=20)


counts_text = f"Positive: {sentiment_counts.get('Positive', 0)} | " \
              f"Neutral: {sentiment_counts.get('Neutral', 0)} | " \
              f"Negative: {sentiment_counts.get('Negative', 0)}"
ax.text(0, -1.35, counts_text, ha='center', fontsize=12, style='italic')

plt.tight_layout()
plt.savefig('outputs/01_worker_sentiment_simple.png', dpi=300, bbox_inches='tight')
print("  Saved: outputs/01_worker_sentiment_simple.png")
plt.close()


print("Creating Visualization 2: What Workers Worry About...")


negative_text = ' '.join(df[df['sentiment'] == 'Negative']['text_clean'].values)
wordcloud_neg = WordCloud(width=1200, height=600, background_color='#ffebee', 
                          colormap='Reds', max_words=80).generate(negative_text)

fig, ax = plt.subplots(figsize=(16, 8))
ax.imshow(wordcloud_neg, interpolation='bilinear')
ax.axis('off')
ax.set_title('CONCERNS: What Workers Worry About with AI\n(Words from worried/negative comments)', 
             fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/02_concerns_wordcloud.png', dpi=300, bbox_inches='tight')
print("  Saved: outputs/02_concerns_wordcloud.png")
plt.close()


print("Creating Visualization 3: What Workers Appreciate...")

positive_text = ' '.join(df[df['sentiment'] == 'Positive']['text_clean'].values)
wordcloud_pos = WordCloud(width=1200, height=600, background_color='#e8f5e9', 
                          colormap='Greens', max_words=80).generate(positive_text)

fig, ax = plt.subplots(figsize=(16, 8))
ax.imshow(wordcloud_pos, interpolation='bilinear')
ax.axis('off')
ax.set_title('APPRECIATION: What Workers Find Useful About AI\n(Words from positive/hopeful comments)', 
             fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/03_appreciation_wordcloud.png', dpi=300, bbox_inches='tight')
print("  Saved: outputs/03_appreciation_wordcloud.png")
plt.close()


print("Creating Visualization 4: Top Concerns...")


concern_keywords = {
    'Job security / Job loss': ['job security', 'replace', 'displacement', 'lose job', 'unemployment'],
    'AI not reliable / Mistakes': ['hallucination', 'wrong', 'mistake', 'error', 'inaccurate', 'fails'],
    'Forced adoption / No choice': ['forced', 'mandate', 'required', 'no option', 'forced down'],
    'Skill gap / Need upskilling': ['upskill', 'learning', 'skill gap', 'keep up', 'training'],
    'Unfairness / Bias': ['bias', 'unfair', 'discrimination', 'bias against', 'unfair advantage'],
    'Trust issues / Transparency': ['trust', 'transparency', 'black box', 'don\'t trust', 'opaque'],
    'Overwork / Pressure': ['overwork', 'pressure', 'stressed', 'burnout', 'workload'],
    'Quality concerns': ['quality', 'bad', 'mediocre', 'poor', 'low quality'],
    'Company greed / Profit motive': ['profit', 'money', 'greedy', 'cut costs', 'layoff'],
    'Privacy concerns': ['privacy', 'surveillance', 'monitor', 'track', 'data']
}

concern_counts = {}
for concern, keywords in concern_keywords.items():
    count = df[df['sentiment'] == 'Negative']['text_clean'].str.contains('|'.join(keywords), case=False).sum()
    concern_counts[concern] = count


concern_df = pd.DataFrame(list(concern_counts.items()), columns=['Concern', 'Count']).sort_values('Count', ascending=True)
concern_df = concern_df[concern_df['Count'] > 0]

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(concern_df['Concern'], concern_df['Count'], color='#e74c3c', edgecolor='black', linewidth=1.2)
ax.set_xlabel('Number of Comments Mentioning This Concern', fontsize=12, fontweight='bold')
ax.set_title('TOP WORKER CONCERNS ABOUT AI\n(What worried/skeptical workers mention)', 
             fontsize=14, fontweight='bold', pad=20)
ax.invert_yaxis()


for i, (bar, value) in enumerate(zip(bars, concern_df['Count'].values)):
    ax.text(value + 0.5, i, str(int(value)), va='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('outputs/04_top_concerns.png', dpi=300, bbox_inches='tight')
print("  Saved: outputs/04_top_concerns.png")
plt.close()


print("Creating Visualization 5: What Workers Like About AI...")

positive_keywords = {
    'Productivity / Efficiency': ['productivity', 'faster', 'efficient', 'quick', 'speed up', 'saves time'],
    'Easier work / Less boring tasks': ['easier', 'boring', 'repetitive', 'automate', 'routine'],
    'Helpful tool / Useful': ['helpful', 'useful', 'assist', 'help', 'support', 'benefits'],
    'Learning / Development': ['learn', 'improve', 'develop', 'growth', 'skill', 'mentor'],
    'Innovation / Creativity': ['creative', 'innovation', 'novel', 'inspire', 'breakthrough'],
    'Better quality / Accuracy': ['better', 'quality', 'accurate', 'improve', 'enhanced'],
    'Cost savings': ['save money', 'cost', 'cheap', 'affordable', 'economical'],
    'Collaboration / Teamwork': ['team', 'collaborate', 'together', 'partnership', 'partner'],
    'Career opportunities': ['opportunity', 'career', 'advancement', 'growth', 'job'],
    'Customer satisfaction': ['customer', 'satisfaction', 'happy', 'experience', 'service']
}

positive_counts = {}
for aspect, keywords in positive_keywords.items():
    count = df[df['sentiment'] == 'Positive']['text_clean'].str.contains('|'.join(keywords), case=False).sum()
    positive_counts[aspect] = count


positive_df = pd.DataFrame(list(positive_counts.items()), columns=['Aspect', 'Count']).sort_values('Count', ascending=True)
positive_df = positive_df[positive_df['Count'] > 0]

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(positive_df['Aspect'], positive_df['Count'], color='#27ae60', edgecolor='black', linewidth=1.2)
ax.set_xlabel('Number of Comments Mentioning This Benefit', fontsize=12, fontweight='bold')
ax.set_title('WHAT WORKERS APPRECIATE ABOUT AI\n(What optimistic/positive workers mention)', 
             fontsize=14, fontweight='bold', pad=20)
ax.invert_yaxis()


for i, (bar, value) in enumerate(zip(bars, positive_df['Count'].values)):
    ax.text(value + 0.5, i, str(int(value)), va='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('outputs/05_what_workers_like.png', dpi=300, bbox_inches='tight')
print("  Saved: outputs/05_what_workers_like.png")
plt.close()


print("Creating Visualization 6: Sentiment by Topic Area...")

fig, ax = plt.subplots(figsize=(13, 7))
sentiment_by_theme = pd.crosstab(df['theme_name'], df['sentiment'], normalize='index') * 100
sentiment_by_theme = sentiment_by_theme[['Positive', 'Neutral', 'Negative']]  
sentiment_by_theme.plot(kind='barh', stacked=True, ax=ax,
                        color=['#27ae60', '#bdc3c7', '#e74c3c'],
                        edgecolor='black', linewidth=1.2)
ax.set_title('SENTIMENT BY TOPIC AREA\n(What are workers discussing?)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Topic / Theme', fontsize=12, fontweight='bold')
ax.legend(title='Sentiment', labels=['Positive', 'Neutral', 'Negative'], 
          bbox_to_anchor=(1.0, 1), loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig('outputs/06_sentiment_by_topic.png', dpi=300, bbox_inches='tight')
print("  Saved: outputs/06_sentiment_by_topic.png")
plt.close()

