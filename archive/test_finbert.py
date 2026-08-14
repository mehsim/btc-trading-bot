import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Add workspace path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import get_news_sentiment

print("=" * 60)
print("TESTING FINBERT DEEP LEARNING SENTIMENT PIPELINE")
print("=" * 60)
print("Fetching latest news headlines and downloading FinBERT (if first run)...")

sentiment, titles = get_news_sentiment()

print("\nResults:")
print(f"Aggregated Sentiment: {sentiment}")
print("\nScanned Headlines:")
for idx, title in enumerate(titles, 1):
    print(f"  {idx}. {title}")
print("=" * 60)
